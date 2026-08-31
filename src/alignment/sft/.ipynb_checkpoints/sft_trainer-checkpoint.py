from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SFTTrainer:

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.scheduler_max_steps,
                    eta_min=config.learning_rate * config.min_lr_ratio,
                )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100
        )

        # --------------------------------------------------
        # Mixed Precision
        # --------------------------------------------------

        self.use_amp = (
            config.use_amp
            and self.device.type == "cuda"
        )

        if self.use_amp:
            self.amp_dtype = getattr(
                torch,
                config.amp_dtype
            )

            self.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=True
            )
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        # --------------------------------------------------
        # Training state
        # --------------------------------------------------

        self.global_step = 0
        self.best_val_loss = float("inf")

        self.checkpoint_dir = Path(
            config.checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ======================================================
    # LOSS
    # ======================================================

    def _compute_loss(self, batch):

        input_ids = batch["input_ids"].to(
            self.device,
            non_blocking=True
        )

        labels = batch["labels"].to(
            self.device,
            non_blocking=True
        )

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            logits = self.model(input_ids)

            loss = self.loss_fn(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                labels.reshape(-1)
            )

        return loss

    # ======================================================
    # VALIDATION
    # ======================================================

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        total_loss = 0.0
        batches = 0

        for batch in self.val_loader:

            loss = self._compute_loss(batch)

            total_loss += loss.item()
            batches += 1

        self.model.train()

        if batches == 0:
            return float("inf")

        return total_loss / batches

    # ======================================================
    # CHECKPOINT
    # ======================================================

    def save_checkpoint(
        self,
        path,
        val_loss=None
    ):

        checkpoint = {
            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "global_step":
                self.global_step,

            "best_val_loss":
                self.best_val_loss,
        }

        if self.use_amp:
            checkpoint["scaler_state_dict"] = (
                self.scaler.state_dict()
            )

        if val_loss is not None:
            checkpoint["val_loss"] = val_loss

        torch.save(
            checkpoint,
            path
        )

    # ======================================================
    # LOAD CHECKPOINT
    # ======================================================

    def load_checkpoint(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        self.global_step = (
            checkpoint["global_step"]
        )
        self.scheduler.last_epoch = self.global_step

        self.best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )

        if (
            self.use_amp
            and "scaler_state_dict" in checkpoint
        ):
            self.scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        print(
            f"Resumed from step "
            f"{self.global_step:,}"
        )

    # ======================================================
    # TRAIN
    # ======================================================

    def train(self):

        self.model.train()

        print("=" * 70)
        print("SFT TRAINING")
        print("=" * 70)

        print(f"Device        : {self.device}")
        print(
            f"Max Steps     : "
            f"{self.config.max_steps:,}"
        )
        print(
            f"Epochs        : "
            f"{self.config.epochs}"
        )
        print(
            f"Learning Rate : "
            f"{self.config.learning_rate}"
        )
        print(
            f"Batch Size    : "
            f"{self.config.batch_size}"
        )
        print(
            f"AMP           : "
            f"{self.use_amp}"
        )

        print("=" * 70)

        steps_per_epoch = len(
            self.train_loader
        )

        print(
            f"Steps / Epoch : "
            f"{steps_per_epoch:,}"
        )

        train_iterator = iter(
            self.train_loader
        )

        running_loss = 0.0
        running_steps = 0

        current_epoch = (
            self.global_step
            // steps_per_epoch
        ) + 1

        while (
            self.global_step
            < self.config.max_steps
            and current_epoch
            <= self.config.epochs
        ):

            try:
                batch = next(
                    train_iterator
                )

            except StopIteration:

                current_epoch += 1

                if current_epoch > self.config.epochs:
                    break

                train_iterator = iter(
                    self.train_loader
                )

                batch = next(
                    train_iterator
                )

            # --------------------------------------------------
            # Forward
            # --------------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            loss = self._compute_loss(
                batch
            )

            # --------------------------------------------------
            # Backward + Optimizer
            # --------------------------------------------------

            if self.use_amp:

                self.scaler.scale(
                    loss
                ).backward()

                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # self.scaler.step(
                #     self.optimizer
                # )

                # self.scaler.update()

                old_scale = self.scaler.get_scale()

                self.scaler.step(self.optimizer)
                self.scaler.update()

                new_scale = self.scaler.get_scale()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                self.optimizer.step()

            # --------------------------------------------------
            # Scheduler
            # --------------------------------------------------

            if new_scale >= old_scale:
                self.scheduler.step()

            # --------------------------------------------------
            # Step bookkeeping
            # --------------------------------------------------

            self.global_step += 1

            running_loss += loss.item()
            running_steps += 1

            # --------------------------------------------------
            # Logging + Validation
            # --------------------------------------------------

            if (
                self.global_step
                % self.config.log_interval
                == 0
            ):

                train_loss = (
                    running_loss
                    / running_steps
                )

                val_loss = self.evaluate()

                lr = (
                    self.optimizer
                    .param_groups[0]["lr"]
                )

                print(
                    f"Epoch [{current_epoch}/"
                    f"{self.config.epochs}] | "
                    f"Step {self.global_step:,} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )

                running_loss = 0.0
                running_steps = 0

                # --------------------------------------------------
                # Best checkpoint
                # --------------------------------------------------

                if val_loss < self.best_val_loss:

                    self.best_val_loss = val_loss

                    best_path = (
                        self.checkpoint_dir
                        / "sft_best.pt"
                    )

                    self.save_checkpoint(
                        best_path,
                        val_loss
                    )

                    print(
                        f"Best checkpoint saved: "
                        f"{best_path}"
                    )

            # --------------------------------------------------
            # Periodic checkpoint
            # --------------------------------------------------

            if (
                    self.config.save_interval > 0
                    and self.global_step
                    % self.config.save_interval
                    == 0
                ):

                checkpoint_path = (
                    self.checkpoint_dir
                    / f"sft_step_"
                    f"{self.global_step}.pt"
                )

                self.save_checkpoint(
                    checkpoint_path
                )

                print(
                    f"Checkpoint saved: "
                    f"{checkpoint_path}"
                )

        # ==================================================
        # FINAL CHECKPOINT
        # ==================================================

        final_path = (
            self.checkpoint_dir
            / "sft_final.pt"
        )

        self.save_checkpoint(
            final_path,
            self.best_val_loss
        )

        print("=" * 70)
        print("SFT TRAINING COMPLETE")
        print("=" * 70)

        print(
            f"Final step      : "
            f"{self.global_step:,}"
        )

        print(
            f"Best val loss   : "
            f"{self.best_val_loss:.4f}"
        )

        print(
            f"Final checkpoint: "
            f"{final_path}"
        )