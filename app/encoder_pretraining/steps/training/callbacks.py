import time

from transformers import TrainerCallback

from utils.slack import notify_training_validation_metric


class ThroughputCallback(TrainerCallback):
    """Print and record training tokens per second at every logging step."""

    def __init__(self, tokens_per_step: int):
        self.tokens_per_step = tokens_per_step
        self.samples: list[float] = []
        self._last_time = None
        self._last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_time = time.monotonic()
        self._last_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        now = time.monotonic()
        steps = state.global_step - self._last_step
        if steps > 0:
            tokens_per_second = steps * self.tokens_per_step / (now - self._last_time)
            self.samples.append(tokens_per_second)
            print(
                f"step={state.global_step}/{state.max_steps} loss={logs['loss']:.4f} "
                f"lr={logs.get('learning_rate', 0):.2e} tokens_per_second={tokens_per_second:,.0f}"
            )
        self._last_time = now
        self._last_step = state.global_step

    def mean_tokens_per_second(self) -> float | None:
        # Skip the first sample, which includes warm-up and data loader start.
        steady = self.samples[1:] or self.samples
        return sum(steady) / len(steady) if steady else None


class SlackEvalCallback(TrainerCallback):
    def __init__(self, workflow_name: str, total_epochs: float, base_model: str):
        self.workflow_name = workflow_name
        self.total_epochs = total_epochs
        self.base_model = base_model

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        epoch = float(state.epoch or 0.0)
        percent = round((epoch - int(epoch)) * 100)
        compact = {
            "eval_loss": f"{metrics['eval_loss']:.4f}" if "eval_loss" in metrics else None,
        }
        notify_training_validation_metric(
            workflow_name=self.workflow_name,
            epoch=epoch,
            total_epochs=self.total_epochs,
            steps=state.global_step,
            metrics={name: value for name, value in compact.items() if value is not None},
            details={"epoch_progress": f"{percent}%", "base_model": self.base_model},
        )
