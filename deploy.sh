#/bin/sh

cd ..
zip -r sagemaker-notebooks.zip sagemaker-notebooks/ -x\
    sagemaker-notebooks/deploy.sh\
    sagemaker-notebooks/README.md\
    sagemaker-notebooks/LICENSE\
    sagemaker-notebooks/.git\*
aws s3 cp sagemaker-notebooks.zip s3://makotosh-tmp/sagemaker-notebooks.zip
rm sagemaker-notebooks.zip