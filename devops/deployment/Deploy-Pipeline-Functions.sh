#!/bin/bash
set -x

# Check if any of the args are empty
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: 'sh $0 (Azure function app name) (Function directory)'"
    exit 1
fi

# Check that the function directory exists
if [ ! -e "$2" ]; then
    echo "Function directory does not exist -- $2"
    exit 1
fi

(cd $2 && func extensions sync)
if [ "$?" -ne 0 ]; then
    echo "Error syncing extensions"
    exit 1
fi

(cd $2 && func azure functionapp publish $1 --force --build-native-deps --no-bundler)
if [ "$?" -ne 0 ]; then
    echo "Error deploying pipeline functions"
    exit 1
fi

