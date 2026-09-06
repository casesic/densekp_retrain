#!/usr/bin/env bash
set -euo pipefail

# 仅下载 rp_aaron_posed_009 所在的官方 clothing texture shard 0。
target_dir="/home/yy/datasets/BEDLAM/clothing_data/clothing_textures"
base_url="https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=clothing_data/clothing_textures"
shard="clothing_textures.0.tar"
limit_bytes=40000000000

urlencode() {
    local LANG=C i char
    for ((i=0; i<${#1}; i++)); do
        char="${1:i:1}"
        [[ "$char" == [a-zA-Z0-9.~-] ]] && printf '%s' "$char" || printf '%%%02X' "'$char"
    done
}

read -r -p "BEDLAM 注册邮箱: " username_raw
read -r -s -p "BEDLAM 密码: " password_raw
printf '\n'
username="$(urlencode "$username_raw")"; password="$(urlencode "$password_raw")"; unset username_raw password_raw
post_data="username=${username}&password=${password}"
header_file="$(mktemp)"; trap 'rm -f "$header_file"; unset password post_data' EXIT
wget --spider --server-response --no-check-certificate --post-data "$post_data" "${base_url}/${shard}" 2>"$header_file" || true
content_length="$(awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*Content-Length:/ {gsub("\r",""); value=$2} END{print value}' "$header_file")"
if [[ "$content_length" =~ ^[0-9]+$ ]]; then
    awk -v n="$content_length" 'BEGIN {printf "TEXTURE_SHARD_SIZE_GB = %.3f\n", n/1000000000}'
    if (( content_length > limit_bytes )); then echo "超过 40 GB 限制，停止下载。" >&2; exit 2; fi
else
    echo "TEXTURE_SHARD_SIZE_GB = UNKNOWN"
fi
mkdir -p "$target_dir"
wget --continue --no-check-certificate --post-data "$post_data" "${base_url}/checksum.xxh128" -O "$target_dir/checksum.xxh128"
wget --continue --no-check-certificate --post-data "$post_data" "${base_url}/${shard}" -O "$target_dir/$shard"
unset password post_data
echo "下载完成: $target_dir/$shard"
