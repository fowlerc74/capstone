BEGIN { FS = "," }
$8 ~ /SOD/ {
    print;
}
