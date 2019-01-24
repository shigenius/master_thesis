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

if [$# -ne 2]; then 
  MESSEGE=$2
  echo "messege:"$MESSEGE
else
  MESSEGE=" "
fi

for i in `seq 10`
do
  python train.py --num_batches 1000 --batch_size 20;
  python evaluate.py --checkpoint_name model.ckpt-1000 > hoge.txt;
  TEXT_=$(cat hoge.txt | grep "num of record")
  TEXT=$MESSEGE$i'\nTrain and evaluation was DONE! \n'$TEXT_

  # post
  curl="curl -X POST --data '{ \
      \"text\": \"${TEXT}\" \
      ,\"username\": \"${USERNAME}\" \
      ,\"link_names\" : ${LINK_NAMES}}' \
      ${URL}"
  eval ${curl}

done

