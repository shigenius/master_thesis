# 設定
URL='https://hooks.slack.com/services/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'
TEXT='test_text'
USERNAME='hoge'
LINK_NAMES='1'

# post
curl="curl -X POST --data '{ \
    \"text\": \"${TEXT}\" \
    ,\"username\": \"${USERNAME}\" \
    ,\"link_names\" : ${LINK_NAMES}}' \
    ${URL}"
eval ${curl}
