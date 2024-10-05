#!/bin/bash

# Step 1: Delete Endpoints
for endpoint in $(aws sagemaker list-endpoints --query "Endpoints[*].EndpointName" --output text); do
    aws sagemaker delete-endpoint --endpoint-name $endpoint
    echo "Deleted endpoint: $endpoint"
done

# Step 2: Delete Endpoint Configurations
for endpoint_config in $(aws sagemaker list-endpoint-configs --query "EndpointConfigs[*].EndpointConfigName" --output text); do
    aws sagemaker delete-endpoint-config --endpoint-config-name $endpoint_config
    echo "Deleted endpoint config: $endpoint_config"
done

# Step 3: Delete Models
for model in $(aws sagemaker list-models --query "Models[*].ModelName" --output text); do
    aws sagemaker delete-model --model-name $model
    echo "Deleted model: $model"
done
