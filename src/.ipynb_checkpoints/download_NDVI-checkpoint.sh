#!/bin/bash

GREP_OPTIONS=''

# Define the target directory
TARGET_DIR="/work/pschluet/green_wave/data"
mkdir -p "$TARGET_DIR"

cookiejar=$(mktemp cookies.XXXXXXXXXX)
netrc=$(mktemp netrc.XXXXXXXXXX)
chmod 0600 "$cookiejar" "$netrc"
function finish {
  rm -rf "$cookiejar" "$netrc"
}

trap finish EXIT
WGETRC="$wgetrc"

prompt_credentials() {
    echo "Enter your Earthdata Login or other provider supplied credentials"
    read -p "Username (t9hzwrtl): " username
    username=${username:-t9hzwrtl}
    read -s -p "Password: " password
    echo "machine urs.earthdata.nasa.gov login $username password $password" >> $netrc
    echo
}

exit_with_error() {
    echo
    echo "Unable to Retrieve Data"
    echo
    echo $1
    echo
    exit 1
}

prompt_credentials

detect_app_approval() {
    approved=$(curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" \
      https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/ \
      -w '\n%{http_code}' | tail -1)
    if [ "$approved" -ne "200" ] && [ "$approved" -ne "301" ] && [ "$approved" -ne "302" ]; then
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below"
    fi
}

setup_auth_curl() {
    status=$(curl -s -z "$(date)" -w '\n%{http_code}' \
      https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/ | tail -1)
    if [[ "$status" -ne "200" && "$status" -ne "304" ]]; then
        detect_app_approval
    fi
}

setup_auth_wget() {
    touch ~/.netrc
    chmod 0600 ~/.netrc
    credentials=$(grep 'machine urs.earthdata.nasa.gov' ~/.netrc)
    if [ -z "$credentials" ]; then
        cat "$netrc" >> ~/.netrc
    fi
}

fetch_urls() {
  if command -v curl >/dev/null 2>&1; then
      setup_auth_curl
      while read -r line; do
        filename="${line##*/}"
        stripped_query_params="${filename%%\?*}"

        curl -f -b "$cookiejar" -c "$cookiejar" -L --netrc-file "$netrc" -g \
          -o "$TARGET_DIR/$stripped_query_params" -- "$line" && echo || \
          exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  elif command -v wget >/dev/null 2>&1; then
      echo "WARNING: Can't find curl, using wget instead."
      setup_auth_wget
      while read -r line; do
        filename="${line##*/}"
        stripped_query_params="${filename%%\?*}"

        wget --load-cookies "$cookiejar" --save-cookies "$cookiejar" \
          --output-document "$TARGET_DIR/$stripped_query_params" \
          --keep-session-cookies -- "$line" && echo || \
          exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  else
      exit_with_error "Error: Could not find a command-line downloader. Please install curl or wget."
  fi
}

fetch_urls <<'EDSCEOF'
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2011017.061.2021181172207/MOD13C1.A2011017.061.2021181172207.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2011001.061.2021180180303/MOD13C1.A2011001.061.2021180180303.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010353.061.2021180024011/MOD13C1.A2010353.061.2021180024011.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010337.061.2021178161250/MOD13C1.A2010337.061.2021178161250.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010321.061.2021178053645/MOD13C1.A2010321.061.2021178053645.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010305.061.2021178054129/MOD13C1.A2010305.061.2021178054129.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010289.061.2021176172249/MOD13C1.A2010289.061.2021176172249.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010273.061.2021176122614/MOD13C1.A2010273.061.2021176122614.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010257.061.2021176063520/MOD13C1.A2010257.061.2021176063520.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010241.061.2021172183418/MOD13C1.A2010241.061.2021172183418.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010225.061.2021169202801/MOD13C1.A2010225.061.2021169202801.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010209.061.2021169031948/MOD13C1.A2010209.061.2021169031948.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010193.061.2021169013901/MOD13C1.A2010193.061.2021169013901.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010177.061.2021168233926/MOD13C1.A2010177.061.2021168233926.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010161.061.2021168204026/MOD13C1.A2010161.061.2021168204026.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010145.061.2021168192116/MOD13C1.A2010145.061.2021168192116.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010129.061.2021161055046/MOD13C1.A2010129.061.2021161055046.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010113.061.2021160034315/MOD13C1.A2010113.061.2021160034315.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010097.061.2021159202331/MOD13C1.A2010097.061.2021159202331.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010081.061.2021159183932/MOD13C1.A2010081.061.2021159183932.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010065.061.2021155075513/MOD13C1.A2010065.061.2021155075513.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010049.061.2021154031032/MOD13C1.A2010049.061.2021154031032.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010033.061.2021153012122/MOD13C1.A2010033.061.2021153012122.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010017.061.2021152222544/MOD13C1.A2010017.061.2021152222544.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2010001.061.2021151070755/MOD13C1.A2010001.061.2021151070755.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009353.061.2021149195056/MOD13C1.A2009353.061.2021149195056.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009337.061.2021148162822/MOD13C1.A2009337.061.2021148162822.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009321.061.2021147021238/MOD13C1.A2009321.061.2021147021238.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009305.061.2021145192731/MOD13C1.A2009305.061.2021145192731.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009289.061.2021144140515/MOD13C1.A2009289.061.2021144140515.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009273.061.2021143214829/MOD13C1.A2009273.061.2021143214829.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009257.061.2021142130626/MOD13C1.A2009257.061.2021142130626.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009241.061.2021141172023/MOD13C1.A2009241.061.2021141172023.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009225.061.2021140113939/MOD13C1.A2009225.061.2021140113939.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009209.061.2021139185306/MOD13C1.A2009209.061.2021139185306.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009193.061.2021138102659/MOD13C1.A2009193.061.2021138102659.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009177.061.2021138002853/MOD13C1.A2009177.061.2021138002853.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009161.061.2021135224526/MOD13C1.A2009161.061.2021135224526.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009145.061.2021135043017/MOD13C1.A2009145.061.2021135043017.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009129.061.2021134025029/MOD13C1.A2009129.061.2021134025029.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009113.061.2021131181137/MOD13C1.A2009113.061.2021131181137.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009097.061.2021126230011/MOD13C1.A2009097.061.2021126230011.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009081.061.2021125000208/MOD13C1.A2009081.061.2021125000208.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009065.061.2021123141838/MOD13C1.A2009065.061.2021123141838.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009049.061.2021122121420/MOD13C1.A2009049.061.2021122121420.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009033.061.2021121090735/MOD13C1.A2009033.061.2021121090735.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009017.061.2021122031450/MOD13C1.A2009017.061.2021122031450.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2009001.061.2021126015435/MOD13C1.A2009001.061.2021126015435.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008353.061.2021114055216/MOD13C1.A2008353.061.2021114055216.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008337.061.2021112103207/MOD13C1.A2008337.061.2021112103207.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008321.061.2021112141302/MOD13C1.A2008321.061.2021112141302.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008305.061.2021110160028/MOD13C1.A2008305.061.2021110160028.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008289.061.2021112163150/MOD13C1.A2008289.061.2021112163150.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008273.061.2021108203909/MOD13C1.A2008273.061.2021108203909.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008257.061.2021107205810/MOD13C1.A2008257.061.2021107205810.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008241.061.2021106161417/MOD13C1.A2008241.061.2021106161417.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008225.061.2021105160132/MOD13C1.A2008225.061.2021105160132.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008209.061.2021104155255/MOD13C1.A2008209.061.2021104155255.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008193.061.2021103172754/MOD13C1.A2008193.061.2021103172754.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008177.061.2021100110127/MOD13C1.A2008177.061.2021100110127.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008161.061.2021099165127/MOD13C1.A2008161.061.2021099165127.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008145.061.2021098155045/MOD13C1.A2008145.061.2021098155045.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008129.061.2021097063202/MOD13C1.A2008129.061.2021097063202.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008113.061.2021097031449/MOD13C1.A2008113.061.2021097031449.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008097.061.2021096215044/MOD13C1.A2008097.061.2021096215044.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008081.061.2021096144227/MOD13C1.A2008081.061.2021096144227.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008065.061.2021089174041/MOD13C1.A2008065.061.2021089174041.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008049.061.2021089044600/MOD13C1.A2008049.061.2021089044600.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008033.061.2021089023319/MOD13C1.A2008033.061.2021089023319.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008017.061.2021089011028/MOD13C1.A2008017.061.2021089011028.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2008001.061.2021088213604/MOD13C1.A2008001.061.2021088213604.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007353.061.2021082045419/MOD13C1.A2007353.061.2021082045419.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007337.061.2021082031703/MOD13C1.A2007337.061.2021082031703.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007321.061.2021079231513/MOD13C1.A2007321.061.2021079231513.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007305.061.2021079055925/MOD13C1.A2007305.061.2021079055925.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007289.061.2021078002042/MOD13C1.A2007289.061.2021078002042.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007273.061.2021076203237/MOD13C1.A2007273.061.2021076203237.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007257.061.2021075044741/MOD13C1.A2007257.061.2021075044741.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007241.061.2021074005358/MOD13C1.A2007241.061.2021074005358.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007225.061.2021073060100/MOD13C1.A2007225.061.2021073060100.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007209.061.2021071125205/MOD13C1.A2007209.061.2021071125205.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007193.061.2021068232403/MOD13C1.A2007193.061.2021068232403.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007177.061.2021068155306/MOD13C1.A2007177.061.2021068155306.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007161.061.2021066091309/MOD13C1.A2007161.061.2021066091309.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007145.061.2021065204142/MOD13C1.A2007145.061.2021065204142.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007129.061.2021065033553/MOD13C1.A2007129.061.2021065033553.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007113.061.2021065023018/MOD13C1.A2007113.061.2021065023018.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007097.061.2021060152120/MOD13C1.A2007097.061.2021060152120.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007081.061.2021059214129/MOD13C1.A2007081.061.2021059214129.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007065.061.2021058045358/MOD13C1.A2007065.061.2021058045358.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007049.061.2021056113334/MOD13C1.A2007049.061.2021056113334.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007033.061.2021053071136/MOD13C1.A2007033.061.2021053071136.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007017.061.2021056003039/MOD13C1.A2007017.061.2021056003039.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2007001.061.2021051211653/MOD13C1.A2007001.061.2021051211653.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006353.061.2021054141755/MOD13C1.A2006353.061.2021054141755.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006337.061.2020279030158/MOD13C1.A2006337.061.2020279030158.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006321.061.2020278042150/MOD13C1.A2006321.061.2020278042150.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006305.061.2020277223149/MOD13C1.A2006305.061.2020277223149.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006289.061.2020275111504/MOD13C1.A2006289.061.2020275111504.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006273.061.2020274230114/MOD13C1.A2006273.061.2020274230114.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006257.061.2020274021243/MOD13C1.A2006257.061.2020274021243.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006241.061.2020274000112/MOD13C1.A2006241.061.2020274000112.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006225.061.2020273015715/MOD13C1.A2006225.061.2020273015715.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006209.061.2020269102319/MOD13C1.A2006209.061.2020269102319.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006193.061.2020267222313/MOD13C1.A2006193.061.2020267222313.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006177.061.2020266081157/MOD13C1.A2006177.061.2020266081157.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006161.061.2020265094152/MOD13C1.A2006161.061.2020265094152.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006145.061.2020263161448/MOD13C1.A2006145.061.2020263161448.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006129.061.2020262212931/MOD13C1.A2006129.061.2020262212931.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006113.061.2020261195035/MOD13C1.A2006113.061.2020261195035.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006097.061.2020260153451/MOD13C1.A2006097.061.2020260153451.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006081.061.2020259060125/MOD13C1.A2006081.061.2020259060125.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006065.061.2020257201037/MOD13C1.A2006065.061.2020257201037.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006049.061.2020256233525/MOD13C1.A2006049.061.2020256233525.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006033.061.2020256080200/MOD13C1.A2006033.061.2020256080200.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006017.061.2020255071801/MOD13C1.A2006017.061.2020255071801.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2006001.061.2020254075640/MOD13C1.A2006001.061.2020254075640.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005353.061.2020252234647/MOD13C1.A2005353.061.2020252234647.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005337.061.2020252090651/MOD13C1.A2005337.061.2020252090651.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005321.061.2020251102948/MOD13C1.A2005321.061.2020251102948.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005305.061.2020250103532/MOD13C1.A2005305.061.2020250103532.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005289.061.2020249124340/MOD13C1.A2005289.061.2020249124340.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005273.061.2020247205336/MOD13C1.A2005273.061.2020247205336.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005257.061.2020246223810/MOD13C1.A2005257.061.2020246223810.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005241.061.2020245225239/MOD13C1.A2005241.061.2020245225239.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005225.061.2020244204307/MOD13C1.A2005225.061.2020244204307.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005209.061.2020242045637/MOD13C1.A2005209.061.2020242045637.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005193.061.2020241155257/MOD13C1.A2005193.061.2020241155257.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005177.061.2020237142012/MOD13C1.A2005177.061.2020237142012.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005161.061.2020236154326/MOD13C1.A2005161.061.2020236154326.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005145.061.2020235131901/MOD13C1.A2005145.061.2020235131901.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005129.061.2020234053645/MOD13C1.A2005129.061.2020234053645.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005113.061.2020233001510/MOD13C1.A2005113.061.2020233001510.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005097.061.2020231122222/MOD13C1.A2005097.061.2020231122222.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005081.061.2020230041155/MOD13C1.A2005081.061.2020230041155.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005065.061.2020228122854/MOD13C1.A2005065.061.2020228122854.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005049.061.2020227215502/MOD13C1.A2005049.061.2020227215502.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005033.061.2020218090948/MOD13C1.A2005033.061.2020218090948.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005017.061.2020216202719/MOD13C1.A2005017.061.2020216202719.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2005001.061.2020216154023/MOD13C1.A2005001.061.2020216154023.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004353.061.2020213214606/MOD13C1.A2004353.061.2020213214606.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004337.061.2020213001847/MOD13C1.A2004337.061.2020213001847.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004321.061.2020212000627/MOD13C1.A2004321.061.2020212000627.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004305.061.2020209195409/MOD13C1.A2004305.061.2020209195409.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004289.061.2020207052827/MOD13C1.A2004289.061.2020207052827.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004273.061.2020207025024/MOD13C1.A2004273.061.2020207025024.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004257.061.2020207015901/MOD13C1.A2004257.061.2020207015901.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004241.061.2020205195315/MOD13C1.A2004241.061.2020205195315.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004225.061.2020199103624/MOD13C1.A2004225.061.2020199103624.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004209.061.2020196130323/MOD13C1.A2004209.061.2020196130323.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004193.061.2020196130737/MOD13C1.A2004193.061.2020196130737.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004177.061.2020196125145/MOD13C1.A2004177.061.2020196125145.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004161.061.2020175002125/MOD13C1.A2004161.061.2020175002125.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004145.061.2020125131337/MOD13C1.A2004145.061.2020125131337.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004129.061.2020125011304/MOD13C1.A2004129.061.2020125011304.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004113.061.2020124124911/MOD13C1.A2004113.061.2020124124911.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004097.061.2020123092049/MOD13C1.A2004097.061.2020123092049.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004081.061.2020122080630/MOD13C1.A2004081.061.2020122080630.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004065.061.2020121071000/MOD13C1.A2004065.061.2020121071000.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004049.061.2020120104636/MOD13C1.A2004049.061.2020120104636.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004033.061.2020120031142/MOD13C1.A2004033.061.2020120031142.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004017.061.2020119152103/MOD13C1.A2004017.061.2020119152103.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2004001.061.2020119041756/MOD13C1.A2004001.061.2020119041756.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003353.061.2020115195056/MOD13C1.A2003353.061.2020115195056.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003337.061.2020114145603/MOD13C1.A2003337.061.2020114145603.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003321.061.2020113213920/MOD13C1.A2003321.061.2020113213920.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003305.061.2020112202012/MOD13C1.A2003305.061.2020112202012.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003289.061.2020111162458/MOD13C1.A2003289.061.2020111162458.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003273.061.2020110033750/MOD13C1.A2003273.061.2020110033750.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003257.061.2020109131417/MOD13C1.A2003257.061.2020109131417.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003241.061.2020109214933/MOD13C1.A2003241.061.2020109214933.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003225.061.2020113220921/MOD13C1.A2003225.061.2020113220921.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003209.061.2020106145334/MOD13C1.A2003209.061.2020106145334.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003193.061.2020098010639/MOD13C1.A2003193.061.2020098010639.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003177.061.2020097062709/MOD13C1.A2003177.061.2020097062709.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003161.061.2020096075221/MOD13C1.A2003161.061.2020096075221.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003145.061.2020094195536/MOD13C1.A2003145.061.2020094195536.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003129.061.2020093195819/MOD13C1.A2003129.061.2020093195819.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003113.061.2020092193057/MOD13C1.A2003113.061.2020092193057.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003097.061.2020090193203/MOD13C1.A2003097.061.2020090193203.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003081.061.2020090181336/MOD13C1.A2003081.061.2020090181336.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003065.061.2020090172441/MOD13C1.A2003065.061.2020090172441.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003049.061.2020090163337/MOD13C1.A2003049.061.2020090163337.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003033.061.2020090153807/MOD13C1.A2003033.061.2020090153807.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003017.061.2020090142402/MOD13C1.A2003017.061.2020090142402.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2003001.061.2020090135446/MOD13C1.A2003001.061.2020090135446.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002353.061.2020084110532/MOD13C1.A2002353.061.2020084110532.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002337.061.2020083173757/MOD13C1.A2002337.061.2020083173757.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002321.061.2020083025518/MOD13C1.A2002321.061.2020083025518.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002305.061.2020083145519/MOD13C1.A2002305.061.2020083145519.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002289.061.2020080085205/MOD13C1.A2002289.061.2020080085205.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002273.061.2020078080255/MOD13C1.A2002273.061.2020078080255.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002257.061.2020077033957/MOD13C1.A2002257.061.2020077033957.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002241.061.2020076045519/MOD13C1.A2002241.061.2020076045519.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002225.061.2020075050922/MOD13C1.A2002225.061.2020075050922.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002209.061.2020077203003/MOD13C1.A2002209.061.2020077203003.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002193.061.2020077195453/MOD13C1.A2002193.061.2020077195453.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002177.061.2020072204542/MOD13C1.A2002177.061.2020072204542.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002161.061.2020072023223/MOD13C1.A2002161.061.2020072023223.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002145.061.2020071131717/MOD13C1.A2002145.061.2020071131717.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002129.061.2020071061024/MOD13C1.A2002129.061.2020071061024.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002113.061.2020070181014/MOD13C1.A2002113.061.2020070181014.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002097.061.2020070102041/MOD13C1.A2002097.061.2020070102041.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002081.061.2020070020642/MOD13C1.A2002081.061.2020070020642.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002065.061.2020069155005/MOD13C1.A2002065.061.2020069155005.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002049.061.2020069092815/MOD13C1.A2002049.061.2020069092815.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002033.061.2020069012131/MOD13C1.A2002033.061.2020069012131.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002017.061.2020068162056/MOD13C1.A2002017.061.2020068162056.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2002001.061.2020068053433/MOD13C1.A2002001.061.2020068053433.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2001353.061.2020067193407/MOD13C1.A2001353.061.2020067193407.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2001337.061.2020067114118/MOD13C1.A2001337.061.2020067114118.hdf
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13C1.061/MOD13C1.A2001321.061.2020067020401/MOD13C1.A2001321.061.2020067020401.hdf
EDSCEOF