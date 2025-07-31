#!/bin/bash

# generates subdirectories RADIOSONDE or DSHIP
# with shipnames as subdirectories. Each ship
# subdirectory contains a further subdirectory
# with the first 6 digits of the md5sum of the
# format string.
#
# The assumptions is that files in directories
# with identical 6-digit-md5sums can be processed
# with the same Fortran programm. 
#
# Takes original directory as input argument
#
# ./GATEsort.sh <inputdir>
#
outfile="summary.txt"
#
# helper functions
# ----------------

simplify_string (){
    
    # erase leading and trailing white space, replace blanks, and remove dots
    local simple_string=$1

    shopt -s extglob
    simple_string="${simple_string##*( )}"
    simple_string="${simple_string%%*( )}"
    shopt -u extglob

    simple_string=$(sed 's/ /_/g' <<< "${simple_string}")
    simple_string=$(sed 's/\.//g' <<< "${simple_string}")

    echo "${simple_string}"
}

erase_blanks (){
    
    # erase leading and trailing white space, replace blanks, and remove dots
    local simple_string=$1

    shopt -s extglob
    simple_string="${simple_string##*( )}"
    simple_string="${simple_string%%*( )}"
    shopt -u extglob

    echo "${simple_string}"
}

# ----------------

# for infile in $1/*
for infile in $(ls $1/* | awk -F\- '{print $NF,$0}' | sort -t- -k1 -n | awk '{print $2}')
do
    echo testing ${infile}
    if test -f ${infile}; then
        echo ${infile}
        n=0

        while read -r line
        do
            n=$((n+1))

            if [ "${line:0:1}" == "0" ] ; then
                echo Type 0 in ${infile}
                break
            elif [ "${line:0:1}" == "4" ] ; then
                echo Type 4 in ${infile}
                break
            elif [ `echo -n "${line:0:1}" | xxd -p` == "ff" ]; then
                echo parity bit in ${infile}
                break
            fi

            # now, consider only files starting with type 1 section
            if [ "${line:0:1}" != "1" ] ; then break; fi

            if [[ $n == 2 ]]; then

                TYPE=${line:3:4}
		PLATFORM=$(simplify_string "${line:15:24}")

                if [[ "${TYPE}" == "BLLN" ]] ; then

                    # Radiosonde data
		    # ---------------

                    read line
                    n=$((n+1))

                    ship=$(simplify_string "${line:15:24}")

                    while (( n < 10 ))
		    do
			read line
			n=$((n+1))
			if (( "${line:79:1}" == "8" )) ; then
			    format1=$(simplify_string "${line:1:76}")
                	elif (( "${line:79:1}" == "9" )) ; then
			    format2=$(simplify_string "${line:1:76}")
			fi
		    done
	    
		    echo "${TYPE}" "${PLATFORM}" on ${ship} >> "$outfile"

		    format_string="${format1}${format2}"
                    md5_format_string=`echo ${format_string} | md5sum | tr -d "-"`

		    this_dir=RADIOSONDE/"${ship}"/"${md5_format_string:0:6}"
		    [[ ! -d "${this_dir}" ]] && mkdir -p "${this_dir}"

                    echo Format $format_string ${md5_format_string:0:6}

                    cp ${infile} "${this_dir}"/.

                    break

                elif [[ "${TYPE}" == "SHIP" ]] ; then

                    # "DSHIP" data
		    # ------------

                    ship="${PLATFORM}"

		    # skip next 4 lines
		    while (( n < 7 ))
		    do
                        read line
			n=$((n+1))
		    done

                    interval=${line:16:7} # omits last two digits ".0"
                    format_type=${line:25:1}

                    if [[ "${interval}" == *".00"* ]]; then
                        num_interval=0
                    elif [[ "${interval}" == *".30"* ]]; then
                        echo "undefined interval " ${interval}
                        num_interval=999
                    fi

                    while (( n < 11 ))
		    do
			read line
			n=$((n+1))
			if ((  "${line:79:1}" == "8" )) ; then
			    format1=$(simplify_string "${line:1:76}")
			elif (( "${line:79:1}" == "9" )) ; then
			    format2=$(simplify_string "${line:1:76}")
			elif ((  "${line:78:2}" == "10" )) ; then
			    format3=$(simplify_string "${line:1:76}")
			fi
		    done
                    # echo ${format_string}
		    format_string="${format1}${format2}${format3}"
		    md5_format_string=`echo ${format_string} | md5sum |tr -d "-"`

                    echo ship data from ${PLATFORM} with interval ${num_interval} ${format_type}

		    this_dir=DSHIP/"${ship}"/"${md5_format_string:0:6}"
		    [[ ! -d "${this_dir}" ]] && mkdir -p "${this_dir}"

                    cp ${infile} "${this_dir}"/.

                    break

                fi

            fi

            if (( $n==25 )) ; then break; fi

        done < ${infile}

    else
        echo ${infile} not found!

    fi # infile

done

echo '...done.'
