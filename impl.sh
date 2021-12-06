declare -a array=()
declare -a array=("struct")

for ((i = 0; i < ${#array[@]}; i++)) {
    a=0
    while [ $a -lt 10 ]
    do
        echo $rec >> log.txt
        python run_link_pred.py --rate $a --type ${array[i]} --dataset cora
        rec=`expr 2 \* $rec`
        a=`expr $a + 1`
    done
    t=`expr $t + 1`
}