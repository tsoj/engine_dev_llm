#!/bin/bash

exe=$1

token=$2

shift
shift

for number in "$@"
do
  command="$exe exportguild -t $token -g $number -f Json -o ./data/discord_json_data/[%G|%C][%g|%c].json"
  echo Command: $command
  $command
done
