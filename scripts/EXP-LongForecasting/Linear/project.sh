if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=180
model_name=DLinear
data_path=electricity/electricity.csv

python run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $data_path \
  --model_id Electricity_$seq_len'_'96 \
  --model $model_name \
  --data other \
  --features S \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 30 \
  --enc_in 128 \
  --des 'Exp' \
  --freq D \
  --step 10 \
  --itr 1 --batch_size 16  --learning_rate 4e-3 \
  # --resume ./checkpoints/Electricity_180_96_DLinear_other_ftS_sl180_ll0_pl10_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/mape_0.10.pth
  # --resume ./checkpoints/Electricity_180_96_DLinear_other_ftS_sl180_ll30_pl90_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/1016_sincos_y_nobias_0.0732.pth

