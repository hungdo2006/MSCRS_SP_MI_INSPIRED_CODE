import argparse
import math
import os
import sys
import time
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from src.rec.data.dataset_dbpedia_inspired import DBpedia ,Co_occurrence ,text_sim ,image_sim
from src.rec.data.dataset_pre_inspired import CRSDataset, CRSDataCollator_mm
from src.rec.eval.evaluate_rec import RecEvaluator
from src.rec.models.model_gpt2 import PromptGPT2forCRS
from src.rec.config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from src.rec.models.model_prompt import MMPrompt_inspired


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='pretrained/pre-trained-inspired', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument("--dataset", type=str, default='inspired', help="A file containing all data.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--max_length", type=int,default=200, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int,default=200)
    parser.add_argument("--entity_max_length", type=int, default=32,help="max entity length in dataset.")
    parser.add_argument("--tokenizer", type=str , default='microsoft/DialoGPT-small')
    parser.add_argument("--text_tokenizer", type=str, default='roberta-base')
    parser.add_argument("--model", type=str, default='microsoft/DialoGPT-small',help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str, default='roberta-base')
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--use_sp", action="store_true", help="Use CMDL-style shared/private representations.")
    parser.add_argument("--use_mi", action="store_true", help="Use CMDL MI disentanglement regularization.")
    parser.add_argument("--mi_weight", type=float, default=0.1)
    parser.add_argument("--mi_sample_size", type=int, default=128)
    parser.add_argument("--mi_club_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=64,help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=8e-4,help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=168)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)
    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    os.makedirs('log', exist_ok=True)
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(config)
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    train_dataset = CRSDataset(
        dataset=args.dataset, split='train', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length
    )
    co = Co_occurrence(dataset=args.dataset, split='train', debug=args.debug ,all_items = kg['item_ids'],entity_max_length=args.entity_max_length,n_entity=kg['num_entities'] ).get_entity_co_info()
    text_simi  = text_sim(pad_entity_id=kg['pad_entity_id']).get_entity_ts_info()
    image_simi = image_sim(pad_entity_id=kg['pad_entity_id']).get_entity_is_info()

    valid_dataset = CRSDataset(
        dataset=args.dataset, split='valid', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length
    )
    test_dataset = CRSDataset(
        dataset=args.dataset, split='test', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length
    )
    data_collator = CRSDataCollator_mm(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        max_length=args.max_length, entity_max_length=args.entity_max_length,
        use_amp=accelerator.mixed_precision == "fp16", debug=args.debug,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        all_items = kg['item_ids'], id_to_idx = text_simi['id_to_idx']
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )


    prompt_encoder = MMPrompt_inspired(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],edge_index_c = co['edge_index_c'],edge_index_i_s = image_simi['edge_index_i_s'],edge_index_t_s = text_simi['edge_index_t_s'], idx_to_id = text_simi['idx_to_id'],
        use_sp=args.use_sp, use_mi=args.use_mi, mi_sample_size=args.mi_sample_size
    ).to(device)

    fix_modules = [model, text_encoder]
    for module in fix_modules:
        module.requires_grad_(False)

    # optim & amp
    modules = [prompt_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    club_prefixes = ("club_zx_zs.", "club_zy_zs.", "club_x_ys_cond.", "club_y_xs_cond.")
    main_named_params = [(n, p) for model in modules for n, p in model.named_parameters() if p.requires_grad and not n.startswith(club_prefixes)]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in main_named_params if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in main_named_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    club_optimizer = None
    if args.use_mi:
        club_params = [p for n, p in prompt_encoder.named_parameters() if p.requires_grad and n.startswith(club_prefixes)]
        if not club_params:
            raise RuntimeError("use_mi=True but CLUB critic parameters were not found.")
        club_optimizer = AdamW(club_params, lr=args.learning_rate, weight_decay=0.0)


    evaluator = RecEvaluator()
    if args.use_mi:
        prompt_encoder, optimizer, club_optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
            prompt_encoder, optimizer, club_optimizer, train_dataloader, valid_dataloader, test_dataloader
        )
    else:
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
            prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
        )
    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # train loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        train_mi_loss = []
        train_club_loss = []
        prompt_encoder.train()
        # train
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds,loss_cl= prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True
            )
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            rec_loss = model(**batch['context'], rec=True).rec_loss / args.gradient_accumulation_steps
            if args.use_mi:
                mi_loss = accelerator.unwrap_model(prompt_encoder).compute_mi_loss()
            else:
                mi_loss = rec_loss.new_zeros(())
            loss = rec_loss + loss_cl * 0.0001 + args.mi_weight * mi_loss
            accelerator.backward(loss)
            train_loss.append(float(loss))
            train_mi_loss.append(float(mi_loss.detach()))

            # optim step
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.use_mi:
                    base_prompt_encoder = accelerator.unwrap_model(prompt_encoder)
                    for _ in range(args.mi_club_steps):
                        club_optimizer.zero_grad()
                        club_learning_loss = base_prompt_encoder.compute_mi_critic_learning_loss()
                        accelerator.backward(club_learning_loss)
                        club_optimizer.step()
                        train_club_loss.append(float(club_learning_loss.detach()))

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')
        if train_mi_loss:
            logger.info(f'epoch {epoch} MI loss {np.mean(train_mi_loss)}')
        if train_club_loss:
            logger.info(f'epoch {epoch} CLUB learning loss {np.mean(train_club_loss)}')

        del train_loss, train_mi_loss, train_club_loss, batch

        # valid
        valid_loss = []
        prompt_encoder.eval()
        for batch in tqdm(valid_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds,loss_cl = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['context'], rec=True)
                valid_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # metric
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        valid_report = {}
        for k, v in report.items():
            if k != 'count':
                valid_report[f'valid/{k}'] = v / report['count']
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # test
        test_loss = []
        prompt_encoder.eval()
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds,loss_cl= prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['context'], rec=True)
                test_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # metric
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        test_report = {}
        for k, v in report.items():
            if k != 'count':
                test_report[f'test/{k}'] = v / report['count']
        test_report['test/loss'] = np.mean(test_loss)
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()


    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    prompt_encoder.save(final_dir)
    logger.info(f'save final model')
