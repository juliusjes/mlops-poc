#!/usr/bin/env bash
ENDPOINT="http://seaweed:8333"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID not set}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY not set}"

if ! aws --endpoint-url "$ENDPOINT" s3api head-bucket \
    --bucket lakefs-bucket 2>/dev/null; then

    aws --endpoint-url "$ENDPOINT" s3api create-bucket \
        --bucket lakefs-bucket \
        #--ownership-controls 'Rules=[{ObjectOwnership=BucketOwnerEnforced}]'
fi

if ! aws --endpoint-url "$ENDPOINT" s3api head-bucket \
    --bucket mlflow-bucket 2>/dev/null; then

    aws --endpoint-url "$ENDPOINT" s3api create-bucket \
        --bucket mlflow-bucket \
        #--ownership-controls 'Rules=[{ObjectOwnership=BucketOwnerEnforced}]'
fi


echo "Buckets ready"
