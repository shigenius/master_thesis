MAX_STEP=10000
EVAL_FREQ=1000
NUM_ROOP=`expr $MAX_STEP / $EVAL_FREQ`

# python train.py --num_batches 1000 --batch_size 20;
# python evaluate.py --checkpoint_name model.ckpt-1000>hoge.txt;
# TEXT=$(cat hoge.txt | grep "num of record")

# push notification to my slack user
# 設定
URL='https://hooks.slack.com/services/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'
# TEXT='Train and evaluation was DONE! \n'$TEXT
USERNAME='train_with_eval.sh'
LINK_NAMES='1'

# # post
# curl="curl -X POST --data '{ \
#     \"text\": \"${TEXT}\" \
#     ,\"username\": \"${USERNAME}\" \
#     ,\"link_names\" : ${LINK_NAMES}}' \
#     ${URL}"
# eval ${curl}

if [ $# -eq 1 ]; then 
  MESSEGE=$1
  echo "messege:"$MESSEGE
else
  MESSEGE=""
fi

for i in `seq ${NUM_ROOP}`
do
  NUM_BATCHES=`expr $i \* $EVAL_FREQ`
  echo "Run Train"$i;
  python train.py --num_batches ${NUM_BATCHES} --batch_size 20;
  echo "Run Test"$i;
  EVAL_LOG_NAME=${MESSEGE}"_"${NUM_BATCHES}".csv"
  python evaluate.py --eval_log_name ${EVAL_LOG_NAME} > hoge.txt;
  TEXT_=$(cat hoge.txt | grep "num of record");
  TEXT=$MESSEGE' '$i'\nTrain and evaluation was DONE! \n'$TEXT_;

  # post
  curl="curl -X POST --data '{ \
      \"text\": \"${TEXT}\" \
      ,\"username\": \"${USERNAME}\" \
      ,\"link_names\" : ${LINK_NAMES}}' \
      ${URL}";
  eval ${curl};

done


