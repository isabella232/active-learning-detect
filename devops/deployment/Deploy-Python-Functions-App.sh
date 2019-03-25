#!/bin/bash

#Automation based from instructions hre: https://github.com/Azure/Azure-Functions/wiki/Azure-Functions-on-Linux-Preview

#Exit on error
set -e

ResourceGroup=$1
ResourceLocation=$2
StorageName=$3
FunctionAppName=$4
AppInsightsName=$5

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ] || [ -z "$7" ] || [ -z "$8" ] || [ -z "$9" ] || [ -z "${10}" ] || [ -z "${11}" ] || [ -z "${12}" ] || [ -z "${13}" ] || [ -z "${14}" ]; then
    echo "Usage: sh $0 (Azure Resource Group Name) (Azure Resource Group Location) (Azure Function Storage Name) (Azure Function App Name) (AppInsightsName) (Storage account) (Source container) (Dest container) (DB Server Name) (DB Username) (DB Password) (DB Name) (Storage Connection String)"
    exit 1
fi

STORAGE_ACCOUNT_NAME="$6"
STORAGE_ACCOUNT_KEY="$7"
SOURCE_CONTAINER_NAME="$8"
DESTINATION_CONTAINER_NAME="$9"
DB_HOST="${10}"
DB_USER="${11}"
DB_PASS="${12}"
DB_NAME="${13}"
STORAGE_CONNECTION_STRING="${14}"

StorageNameLength=${#StorageName}
if [ $StorageNameLength -lt 3 -o $StorageNameLength -gt 24 ]; then
    echo "Storage account name must be between 3 and 24 characters in length."
    exit 1
fi

if [[ "$StorageName" != *[a-z0-9]* ]]; then
    echo "Storage account name must use numbers and lower-case letters only"
    exit 1
fi

# See http://jmespath.org/tutorial.html for querying
filtered_output=$(az extension list --query "[?name=='functionapp'].name")

if [[ $filtered_output =~ "functionapp" ]];
then
    echo
    echo "Removing existng Azure CLI extension..."
    az extension remove -n functionapp
fi

TempDownloadLocation="/tmp/functionapp-0.0.2-py2.py3-none-any.whl"

echo
echo "Downloading Azure CLI extension for the Azure Functions Linux Consumption preview"
echo
curl -s -o $TempDownloadLocation "https://functionscdn.azureedge.net/public/docs/functionapp-0.0.2-py2.py3-none-any.whl"

echo
echo "Installing Azure CLI extension for the Azure Functions Linux Consumption preview"
echo
az extension add --yes --source $TempDownloadLocation

echo
echo "Create a resource group (if it does not exist for the current subscription)"
echo
az group create -n $ResourceGroup -l $ResourceLocation

echo
echo "Create a storage account for the function (if it does not exist for the current subscription)"
echo
az storage account create -n $StorageName -l $ResourceLocation -g $ResourceGroup --sku Standard_LRS

echo
echo "Create a function app (if it does not exist for the current subscription)"
echo
az functionapp createpreviewapp -n $FunctionAppName -g $ResourceGroup -l $ResourceLocation-s $StorageName --runtime python --is-linux

echo
echo "Retrieving App Insights Id for $AppInsightsName"
echo
AppInsightsKey=$(az resource show -g $ResourceGroup -n $AppInsightsName --resource-type "Microsoft.Insights/components" --query properties.InstrumentationKey)

#Remove double quotes
AppInsightsKey=$(sed -e 's/^"//' -e 's/"$//' <<<"$AppInsightsKey")
STORAGE_ACCOUNT_KEY=$(sed -e 's/^"//' -e 's/"$//' <<<"$STORAGE_ACCOUNT_KEY")
STORAGE_CONNECTION_STRING=$(sed -e 's/^"//' -e 's/"$//' <<<"$STORAGE_CONNECTION_STRING")

echo
echo "Setting application setting on $FunctionAppName"
echo
az functionapp config appsettings set --name $FunctionAppName --resource-group $ResourceGroup \
    --settings "APPINSIGHTS_INSTRUMENTATIONKEY=$AppInsightsKey" \
                "DB_HOST=$DB_HOST" \
                "DB_USER=$DB_USER" \
                "DB_NAME=$DB_NAME" \
                "DB_PASS=$DB_PASS" \
                "STORAGE_ACCOUNT_NAME=$STORAGE_ACCOUNT_NAME" \
                "STORAGE_ACCOUNT_KEY=$STORAGE_ACCOUNT_KEY" \
                "SOURCE_CONTAINER_NAME=$SOURCE_CONTAINER_NAME" \
                "DESTINATION_CONTAINER_NAME=$DESTINATION_CONTAINER_NAME" \
                "STORAGE_CONNECTION_STRING=$STORAGE_CONNECTION_STRING"
