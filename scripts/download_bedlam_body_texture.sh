#!/usr/bin/env bash
set -euo pipefail

# 仅下载 BEDLAM v1 官方 body texture archive；不会解压或下载其他资源。
readonly bedlam_root="${BEDLAM_ROOT:-$HOME/datasets/BEDLAM}"
readonly target_dir="$bedlam_root/body_textures"
readonly filename="bedlam_body_textures_meshcapade.zip"
readonly url="https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=${filename}"
readonly expected_md5="4963ae5f8d7a5f77857313acc7ef4846"
readonly expected_xxh128="31cca341cde34aea2c73f5ec1ec9b2ae"
readonly limit_bytes=40000000000

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
username="$(urlencode "$username_raw")"
password="$(urlencode "$password_raw")"
unset username_raw password_raw
post_data="username=${username}&password=${password}"
unset username password

header_file="$(mktemp)"
listing_file="$(mktemp)"
cleanup() {
    rm -f "$header_file" "$listing_file"
    unset post_data
}
trap cleanup EXIT INT TERM

# 认证后预检 Content-Length；取最后一个响应头中的值。
wget --spider --server-response --no-check-certificate \
    --post-data "$post_data" "$url" 2>"$header_file" || true
content_length="$(awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*Content-Length:/ {gsub("\r",""); value=$2} END{print value}' "$header_file")"
if [[ "$content_length" =~ ^[0-9]+$ ]]; then
    awk -v n="$content_length" 'BEGIN {printf "BODY_TEXTURE_SIZE_GB = %.3f\n", n/1000000000}'
    if (( content_length > limit_bytes )); then
        echo "Content-Length 超过 40 GB，停止下载。" >&2
        exit 2
    fi
else
    echo "BODY_TEXTURE_SIZE_GB = UNKNOWN"
fi

mkdir -p "$target_dir"
wget --continue --no-check-certificate --post-data "$post_data" \
    "$url" -O "$target_dir/$filename"
unset post_data

actual_bytes="$(stat -c '%s' "$target_dir/$filename")"
awk -v n="$actual_bytes" 'BEGIN {printf "BODY_TEXTURE_ACTUAL_BYTES = %.0f\nBODY_TEXTURE_ACTUAL_SIZE_GB = %.3f\n", n, n/1000000000}'

actual_md5="$(md5sum "$target_dir/$filename" | awk '{print $1}')"
if [[ "$actual_md5" != "$expected_md5" ]]; then
    echo "MD5_CHECK = FAIL" >&2
    exit 3
fi
echo "MD5_CHECK = PASS"

if command -v xxhsum >/dev/null 2>&1 && xxhsum --help 2>&1 | grep -q 'XXH128'; then
    actual_xxh128="$(xxhsum -H128 "$target_dir/$filename" | awk '{print $1}')"
    if [[ "${actual_xxh128,,}" != "$expected_xxh128" ]]; then
        echo "XXH128_CHECK = FAIL" >&2
        exit 4
    fi
    echo "XXH128_CHECK = PASS"
else
    echo "XXH128_CHECK = NOT_SUPPORTED"
fi

# 只列出目录，不解压 archive。
unzip -l "$target_dir/$filename" | tee "$listing_file"

male_count="$(awk 'BEGIN{IGNORECASE=1} /(^|[/_])male([/_]|$)|skin_m_/ {n++} END{print n+0}' "$listing_file")"
female_count="$(awk 'BEGIN{IGNORECASE=1} /(^|[/_])female([/_]|$)|skin_f_/ {n++} END{print n+0}' "$listing_file")"
eye_count="$(awk 'BEGIN{IGNORECASE=1} /eye/ {n++} END{print n+0}' "$listing_file")"
body_count="$(awk 'BEGIN{IGNORECASE=1} /skin|body/ {n++} END{print n+0}' "$listing_file")"
echo "MALE_TEXTURE_MEMBERS = $male_count"
echo "FEMALE_TEXTURE_MEMBERS = $female_count"
echo "EYE_RELATED_MEMBERS = $eye_count"
echo "BODY_RELATED_MEMBERS = $body_count"
echo "ARCHIVE_AUDIT = PASS"
echo "下载完成: $target_dir/$filename"
