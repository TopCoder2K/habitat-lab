#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
import json

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from habitat import logger
from habitat.datasets.utils import VocabDict
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.data.data import EQADataset
from habitat_baselines.il.metrics import VqaMetric
from habitat_baselines.il.models.models import (
    VqaLstmCnnAttentionModel, build_mdetr
)
from habitat_baselines.il.models.mdetr import apply_dt_fixup, load_mdetr_ckpt
from habitat_baselines.utils.common import img_bytes_2_np_array
from habitat_baselines.utils.visualizations.utils import save_vqa_image_results


@baseline_registry.register_trainer(name="vqa")
class VQATrainer(BaseILTrainer):
    r"""Trainer class for VQA model used in EmbodiedQA (Das et. al.; CVPR 2018)
    Paper: https://embodiedqa.org/paper.pdf.
    """
    supported_tasks = ["VQA-v0"]

    def __init__(self, config=None):
        super().__init__(config)

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if config is not None:
            logger.info(f"config: {config}")
        logger.info(f"VQATrainer's DEVICE: {self.device}")

    def _make_results_dir(self):
        r"""Makes directory for saving VQA eval results."""
        dir_name = self.config.RESULTS_DIR.format(split="val")
        os.makedirs(dir_name, exist_ok=True)

    def _save_vqa_results(
        self,
        ckpt_idx: int,
        episode_ids: torch.Tensor,
        questions: Tuple[str],
        images: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_answers: torch.Tensor,
        q_vocab_dict: VocabDict,
        ans_vocab_dict: VocabDict,
    ) -> None:

        r"""For saving VQA results.
        Args:
            ckpt_idx: idx of checkpoint being evaluated
            episode_ids: episode ids of batch
            questions: input questions to model (in natural form)
            images: images' tensor containing input frames
            pred_scores: model prediction scores
            gt_answers: ground truth answers
            q_vocab_dict: Question VocabDict
            ans_vocab_dict: Answer VocabDict

        Returns:
            None
        """
        episode_id = episode_ids[0].item()
        question = questions[0]
        images = images[0]
        gt_answer = gt_answers[0]
        scores = pred_scores[0]

        # q_string = q_vocab_dict.token_idx_2_string(question)

        _, index = scores.max(0)
        pred_answer = sorted(ans_vocab_dict.word2idx_dict.keys())[index]
        gt_answer = sorted(ans_vocab_dict.word2idx_dict.keys())[gt_answer]

        logger.info("Question: {}".format(question))
        logger.info("Predicted answer: {}".format(pred_answer))
        logger.info("Ground-truth answer: {}".format(gt_answer))

        result_path = self.config.RESULTS_DIR.format(
            split=self.config.TASK_CONFIG.DATASET.SPLIT
        )

        result_path = os.path.join(
            result_path, "ckpt_{}_{}_image.jpg".format(ckpt_idx, episode_id)
        )

        save_vqa_image_results(
            images, question, pred_answer, gt_answer, result_path
        )

    def train(self) -> None:
        r"""Main method for training VQA (Answering) model of EQA.

        Returns:
            None
        """
        config = self.config

        # env = habitat.Env(config=config.TASK_CONFIG)

        vqa_dataset = (
            EQADataset(
                config,
                input_type="vqa",
                num_frames=config.IL.VQA.num_frames,
            )
            .shuffle(1000)
            .to_tuple(
                "episode_id",
                "question",
                "answer",
                *["{0:0=3d}.jpg".format(x) for x in range(0, 5)],
            )
            .map(img_bytes_2_np_array)
        )

        train_loader = DataLoader(
            vqa_dataset, batch_size=config.IL.VQA.batch_size
        )

        logger.info("train_loader has {} samples".format(len(vqa_dataset)))

        q_vocab_dict, ans_vocab_dict = vqa_dataset.get_vocab_dicts()

        if config.IL.VQA.model == "lstm_based":
            # TODO: incorporate tokenization inside the lstm-based model
            model_kwargs = {
                "q_vocab": q_vocab_dict.word2idx_dict,
                "ans_vocab": ans_vocab_dict.word2idx_dict,
                "freeze_encoder": config.IL.VQA.freeze_encoder,
                "eqa_cnn_pretrain_ckpt_path": config.EQA_CNN_PRETRAIN_CKPT_PATH
            }
            model = VqaLstmCnnAttentionModel(**model_kwargs)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(config.IL.VQA.lr),
            )
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda lr: lr  # lr isn't changed
            )
        else:
            model, _ = build_mdetr(
                config, len(ans_vocab_dict.word2idx_dict)
            )
            if config.TRAIN_CKPT_PATH:
                checkpoint_path = config.TRAIN_CKPT_PATH
                logger.info(f"Loading MDETR from {checkpoint_path}")
                model = load_mdetr_ckpt(model, checkpoint_path)
            if config.IL.MDETR.apply_dt_fixup:
                logger.info("Applying DT-Fixup to MDETR")
                model = apply_dt_fixup(self.device, model, train_loader)

            param_dicts = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "backbone" not in n and "text_encoder" not in n
                        and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "backbone" in n and p.requires_grad
                    ],
                    "lr": float(config.IL.CNN.lr_backbone),
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "text_encoder" in n and p.requires_grad
                    ],
                    "lr": float(config.IL.TRANSFORMER.text_encoder_lr),
                },
            ]
            optimizer = torch.optim.AdamW(
                param_dicts, lr=float(config.IL.MDETR.lr),
                weight_decay=float(config.IL.VQA.weight_decay)
            )
            lr_scheduler_name = config.IL.VQA.lr_scheduler
            if lr_scheduler_name == "CosineAnnealingLR":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=200, eta_min=float(config.IL.MDETR.lr)/10
                )
            elif lr_scheduler_name == "None":
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda lr: lr  # lr isn't changed
                )
            else:
                assert False, f"Unknown lr_scheduler {lr_scheduler_name}"

        loss_fn = torch.nn.CrossEntropyLoss()

        metrics = VqaMetric(
            info={"split": "train"},
            metric_names=[
                "loss",
                "accuracy",
                "mean_rank",
                "mean_reciprocal_rank",
            ],
            log_json=os.path.join(config.OUTPUT_LOG_DIR, "train.json"),
        )

        t, epoch = 0, 1

        avg_loss = 0.0
        avg_accuracy = 0.0
        avg_mean_rank = 0.0
        avg_mean_reciprocal_rank = 0.0

        logger.info(model)
        model.train().to(self.device)

        if config.IL.CNN.freeze_encoder:
            model.cnn.eval() if config.IL.VQA.model == "lstm_based" \
                else model.backbone.eval()

        with TensorboardWriter(
            config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            while epoch <= config.IL.VQA.max_epochs:
                start_time = time.time()
                for batch in train_loader:
                    t += 1
                    episode_ids, questions, answers, frame_queue = batch
                    optimizer.zero_grad()
                    # cur_episode_id = episode_ids[0]
                    # print(cur_episode_id, vqa_dataset.episodes[cur_episode_id].question)
                    # answer_text = vqa_dataset.episodes[cur_episode_id].question.answer_text
                    # answer_token = ans_vocab_dict.word2idx(answer_text)
                    # print(questions[0], answers[0], answer_token)
                    # assert answers[0] == answer_token, "answer tokens are not equal!"
                    # print(q_vocab_dict.token_idx_2_string(questions[0]), ans_vocab_dict.token_idx_2_string([answers[0], ]), ans_vocab_dict.token_idx_2_string([answer_token]))
                    # print("DEBUG\n", sorted(ans_vocab_dict.word2idx_dict.keys())[answer_token], "\nDEBUG\n")
                    # raise RuntimeError

                    # questions = questions.to(self.device)
                    answers = answers.to(self.device)
                    frame_queue = frame_queue.to(self.device)

                    if config.IL.VQA.model == "lstm_based":
                        scores, _ = model(frame_queue, questions)
                    else:
                        memory_cache = model(
                            frame_queue, questions, encode_and_save=True
                        )
                        outputs, _ = model(
                            frame_queue, questions, encode_and_save=False,
                            memory_cache=memory_cache
                        )
                        scores = outputs["pred_answer"]

                    loss = loss_fn(scores, answers)

                    # update metrics
                    accuracy, ranks = metrics.compute_ranks(
                        scores.data.cpu(), answers
                    )
                    metrics.update([loss.item(), accuracy, ranks, 1.0 / ranks])

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()  # Especially was put here

                    (
                        metrics_loss,
                        accuracy,
                        mean_rank,
                        mean_reciprocal_rank,
                    ) = metrics.get_stats()

                    avg_loss += metrics_loss
                    avg_accuracy += accuracy
                    avg_mean_rank += mean_rank
                    avg_mean_reciprocal_rank += mean_reciprocal_rank

                    if t % config.LOG_INTERVAL == 0:
                        logger.info("Epoch: {}".format(epoch))
                        logger.info(metrics.get_stat_string())

                        writer.add_scalar("loss", metrics_loss, t)
                        writer.add_scalar("accuracy", accuracy, t)
                        writer.add_scalar("mean_rank", mean_rank, t)
                        writer.add_scalar(
                            "mean_reciprocal_rank", mean_reciprocal_rank, t
                        )

                        metrics.dump_log()

                # Dataloader length for IterableDataset doesn't take into
                # account batch size for Pytorch v < 1.6.0
                num_batches = math.ceil(
                    len(vqa_dataset) / config.IL.VQA.batch_size
                )

                avg_loss /= num_batches
                avg_accuracy /= num_batches
                avg_mean_rank /= num_batches
                avg_mean_reciprocal_rank /= num_batches

                end_time = time.time()
                time_taken = "{:.1f}".format((end_time - start_time) / 60)

                logger.info(
                    "Epoch {} completed. Time taken: {} minutes.".format(
                        epoch, time_taken
                    )
                )

                logger.info("Average loss: {:.2f}".format(avg_loss))
                logger.info("Average accuracy: {:.2f}".format(avg_accuracy))
                logger.info("Average mean rank: {:.2f}".format(avg_mean_rank))
                logger.info(
                    "Average mean reciprocal rank: {:.2f}".format(
                        avg_mean_reciprocal_rank
                    )
                )

                print("-----------------------------------------")

                self.save_checkpoint(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    "epoch_{}.ckpt".format(epoch)
                )

                epoch += 1

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        config = self.config

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        config.freeze()

        vqa_dataset = (
            EQADataset(
                config,
                input_type="vqa",
                num_frames=config.IL.VQA.num_frames,
            )
            .shuffle(1000)
            .to_tuple(
                "episode_id",
                "question",
                "answer",
                *["{0:0=3d}.jpg".format(x) for x in range(0, 5)],
            )
            .map(img_bytes_2_np_array)
        )

        eval_loader = DataLoader(
            vqa_dataset, batch_size=config.IL.VQA.batch_size
        )

        logger.info("eval_loader has {} samples".format(len(vqa_dataset)))

        q_vocab_dict, ans_vocab_dict = vqa_dataset.get_vocab_dicts()

        if config.IL.VQA.model == "lstm_based":
            # TODO: incorporate tokenization inside the lstm-based model
            model_kwargs = {
                "q_vocab": q_vocab_dict.word2idx_dict,
                "ans_vocab": ans_vocab_dict.word2idx_dict,
                "eqa_cnn_pretrain_ckpt_path": config.EQA_CNN_PRETRAIN_CKPT_PATH
            }
            model = VqaLstmCnnAttentionModel(**model_kwargs)
            state_dict = torch.load(
                checkpoint_path, map_location={"cuda:0": "cpu"}
            )
            model.load_state_dict(state_dict)
            model.cnn.eval()
            model.to(self.device)
        else:
            model, _ = build_mdetr(
                config, len(ans_vocab_dict.word2idx_dict)
            )
            logger.info(f"Loading MDETR from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            model.eval()
            model.to(self.device)

            # with open("data/eqa/vqa/vqa2_answer2id.json", "r") as f:
            #     answer2id = json.load(f)
            # id2answer = {n_w: w for w, n_w in answer2id.items()}

        loss_fn = torch.nn.CrossEntropyLoss()

        t = 0

        avg_loss = 0.0
        avg_accuracy = 0.0
        avg_mean_rank = 0.0
        avg_mean_reciprocal_rank = 0.0

        metrics = VqaMetric(
            info={"split": "val"},
            metric_names=[
                "loss",
                "accuracy",
                "mean_rank",
                "mean_reciprocal_rank",
            ],
            log_json=os.path.join(config.OUTPUT_LOG_DIR, "eval.json"),
        )

        # Intersection calculation
        # gen_word_cnt = {}
        # for word in ans_vocab_dict.word2idx_dict.keys():
        #     if word in answer2id and word not in gen_word_cnt:
        #         gen_word_cnt[word] = 1
        # print(f"mdetr has: {len(answer2id)}, eqa has: {len(ans_vocab_dict.word2idx_dict)}, general intersection: {len(gen_word_cnt)}")

        # eval_word_cnt = {}
        with torch.no_grad():
            # accuracy_total = 0.
            # accuracy_common = 0.
            # common_ans_cnt = 0
            for batch in eval_loader:
                t += 1
                episode_ids, questions, answers, frame_queue = batch
                # questions = questions.to(self.device)
                answers = answers.to(self.device)
                frame_queue = frame_queue.to(self.device)

                if config.IL.VQA.model == "lstm_based":
                    scores, _ = model(frame_queue, questions)
                else:
                    memory_cache = model(
                        frame_queue, questions, encode_and_save=True
                    )
                    outputs, _ = model(
                        frame_queue, questions, encode_and_save=False,
                        memory_cache=memory_cache
                    )
                    scores = outputs["pred_answer"]
                    # answer_ids = torch.argmax(scores, dim=-1)
                    # accuracy = 0.
                    # for i, ind in enumerate(answer_ids):
                    #     answer_text = vqa_dataset.episodes[episode_ids[i]].question.answer_text
                    #     accuracy += (id2answer[ind.item()] == answer_text)
                    #     accuracy_total += (id2answer[ind.item()] == answer_text)
                    #     if answer_text in answer2id:
                    #         accuracy_common += (id2answer[ind.item()] == answer_text)
                    #         common_ans_cnt += 1
                    #         if answer_text not in eval_word_cnt:
                    #             eval_word_cnt[answer_text] = 1
                    # accuracy /= len(answer_ids)

                loss = loss_fn(scores, answers)

                accuracy, ranks = metrics.compute_ranks(
                    scores.data.cpu(), answers
                )
                metrics.update([loss.item(), accuracy, ranks, 1.0 / ranks])

                (
                    metrics_loss,
                    accuracy,
                    mean_rank,
                    mean_reciprocal_rank,
                ) = metrics.get_stats(mode=0)

                avg_loss += metrics_loss
                avg_accuracy += accuracy
                avg_mean_rank += mean_rank
                avg_mean_reciprocal_rank += mean_reciprocal_rank

                if t % config.LOG_INTERVAL == 0:
                    logger.info(metrics.get_stat_string(mode=0))
                    metrics.dump_log()

                if (
                    config.EVAL_SAVE_RESULTS
                    and t % config.EVAL_SAVE_RESULTS_INTERVAL == 0
                ):

                    self._save_vqa_results(
                        checkpoint_index,
                        episode_ids,
                        questions,
                        frame_queue,
                        scores,
                        answers,
                        q_vocab_dict,
                        ans_vocab_dict,
                    )

        num_batches = math.ceil(len(vqa_dataset) / config.IL.VQA.batch_size)

        avg_loss /= num_batches
        avg_accuracy /= num_batches
        avg_mean_rank /= num_batches
        avg_mean_reciprocal_rank /= num_batches
        # print("Correct answers total:", accuracy_total, "correct common:", accuracy_common, "total common", common_ans_cnt)
        # print(f"Eval dict intersection: {len(eval_word_cnt)}")
        # accuracy_total /= len(vqa_dataset)
        # accuracy_common /= common_ans_cnt

        writer.add_scalar("avg val loss", avg_loss, checkpoint_index)
        writer.add_scalar("avg val accuracy", avg_accuracy, checkpoint_index)
        writer.add_scalar("avg val mean rank", avg_mean_rank, checkpoint_index)
        writer.add_scalar(
            "avg val mean reciprocal rank",
            avg_mean_reciprocal_rank,
            checkpoint_index,
        )

        logger.info("Average loss: {:.2f}".format(avg_loss))
        logger.info("Average accuracy: {:.4f}".format(avg_accuracy))
        # logger.info("Average total accuracy: {:.4f}".format(accuracy_total))
        # logger.info("Average common accuracy: {:.4f}".format(accuracy_common))
        logger.info("Average mean rank: {:.2f}".format(avg_mean_rank))
        logger.info(
            "Average mean reciprocal rank: {:.2f}".format(
                avg_mean_reciprocal_rank
            )
        )
