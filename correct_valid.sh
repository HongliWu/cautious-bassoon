#!/bin/bash
echo "python src/train_runner_correct_valid.py --action train --model se_resnext101_dr0.75_512_aug21"
python src/train_runner_correct_valid.py --action train --model se_resnext101_dr0.75_512_aug21

echo "python src/train_runner_correct_valid.py --action train --model res2net101_v1b_26w_4s_dr0.75_512_aug21"
python src/train_runner_correct_valid.py --action train --model res2net101_v1b_26w_4s_dr0.75_512_aug21

echo "python src/train_runner_correct_valid.py --action train --model res2net101_v1b_26w_4s_dr0.75_512"
python src/train_runner_correct_valid.py --action train --model res2net101_v1b_26w_4s_dr0.75_512

echo "python src/train_runner_correct_valid.py --action train --model res2net101_v1b_26w_4s_dr0.75_512 --run QFL"
python src/train_runner_correct_valid.py --action train --model res2net101_v1b_26w_4s_dr0.75_512 --run QFL

echo "python src/train_runner_correct_valid.py --action train  --run QFL_2"
python src/train_runner_correct_valid.py --action train  --run QFL_2 

echo "python src/train_runner_correct_valid.py --action train --run QFL"
python src/train_runner_correct_valid.py --action train --run QFL

echo "python src/train_runner_correct_valid.py --action train --run reg_ciou"
python src/train_runner_correct_valid.py --action train --run reg_ciou

echo "python src/train_runner_correct_valid.py --action train --run kl_loss"
python src/train_runner_correct_valid.py --action train --run kl_loss

:set fileformat=unix

