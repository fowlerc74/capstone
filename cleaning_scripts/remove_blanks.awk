BEGIN { FS = "," }
NR > 1 {
    for (i=1;i<=NF;i++) {
        if ($i !~ /""/) { 
            print;
        }
    }
}