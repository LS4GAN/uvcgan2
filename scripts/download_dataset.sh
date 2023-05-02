#!/usr/bin/env bash

DATADIR="${UVCGAN2_DATA:-data}"

declare -A URL_LIST=(
    [selfie2anime]="https://www.dropbox.com/s/9lz6gwwwyyxpdnn/selfie2anime.zip"
    [male2female]="https://cgm.technion.ac.il/Computer-Graphics-Multimedia/CouncilGAN/DataSet/celeba_male2female.zip"
    [glasses]="https://cgm.technion.ac.il/Computer-Graphics-Multimedia/CouncilGAN/DataSet/celeba_glasses.zip"
    [celeba_hq]="https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip"
    [afhq]="https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip"
)

declare -A CHECKSUMS=(
    [selfie2anime]="2e8fe7563088971696d29af9f5153772733ac879c155c709b1aad741735ad7bc"
    [male2female]="97178617b01af691b68f0b97de142c6be3331803b79906666fc9ab76f454a18e"
    [glasses]="f4f141469fb8955822042d0999adcc81ec40db875c9bc930b733915b2089613f"
    [celeba_hq]="f56b0d8c505aa01ec4792f8ee2fc2ca128e2cc669277438c0b27d5d69b7e4514"
    [afhq]="7f63dcc14ef58c0e849b59091287e1844da97016073aac20403ae6c6132b950f"
)

die ()
{
    echo "${*}"
    exit 1
}

usage ()
{
    cat <<EOF
USAGE: download_dataset.sh DATASET

where DATASET is one of selfie2anime, male2female, glasses, celeba_all,
celeba_hq, or afhq.
EOF

    if [[ $# -gt 0 ]]
    then
        die "${*}"
    else
        exit 0
    fi
}

exec_or_die ()
{
    "${@}" || die "Failed to execute: '${*}'"
}

calc_sha256_hash ()
{
    local path="${1}"
    sha256sum "${path}" | cut -d ' ' -f 1 | tr -d '\n'
}

download_archive ()
{
    local url="${1}"
    local archive="${2}"
    local checksum="${3}"

    exec_or_die mkdir -p "${DATADIR}"

    local path="${DATADIR}/${archive}"

    if [[ ! -e "${DATADIR}/${archive}" ]]
    then
        exec_or_die wget --no-check-certificate \
            "${url}" --output-document "${path}"
    fi

    if [[ -n "${checksum}" ]]
    then
        # shellcheck disable=SC2155
        local test_csum="$(calc_sha256_hash "${path}")"

        if [[ "${test_csum}" == "${checksum}" ]]
        then
            echo " - Checksum valid"
        else
            die "Checksum mismatch for '${path}' ${test_csum} vs ${checksum}"
        fi
    fi
}

download_and_extract_zip ()
{
    local url="${1}"
    local zip="${2}"
    local checksum="${3}"

    download_archive  "${url}" "${zip}" "${checksum}"
    exec_or_die unzip "${DATADIR}/${zip}" -d "${DATADIR}"

    # exec_or_die rm "${dst}/${zip}"

    echo " - Dataset is unpacked to '${path}'"
}

check_dset_exists ()
{
    local path="${1}"

    if [[ -e "${path}" ]]
    then

        read -r -p "Dataset '${path}' exists. Overwrite? [yN]: " ret
        case "${ret}" in
            [Yy])
                exec_or_die rm -rf "${path}"
                ;;
            *)
                exit 0
                ;;
        esac
    fi
}

download_anime2selfie ()
{
    local url="${URL_LIST["selfie2anime"]}"
    local zip="selfie2anime.zip"
    local path="${DATADIR}/selfie2anime"

    check_dset_exists "${path}"

    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[selfie2anime]}"

    # CouncilGAN mangled dataset
    exec_or_die mv "${path}/trainA" "${path}/tmp"
    exec_or_die mv "${path}/trainB" "${path}/trainA"
    exec_or_die mv "${path}/tmp"    "${path}/trainB"

    exec_or_die mv "${path}/testA" "${path}/tmp"
    exec_or_die mv "${path}/testB" "${path}/testA"
    exec_or_die mv "${path}/tmp"   "${path}/testB"
}

download_male2female ()
{
    local url="${URL_LIST["male2female"]}"
    local zip="male2female.zip"
    local path="${DATADIR}/celeba_male2female"

    check_dset_exists "${path}"
    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[male2female]}"
}

move_files ()
{
    local dst="${1}"
    shift
    local src=( "${@}" )

    # NOTE: too many mv calls. Maybe optimize with xargs
    exec_or_die find "${src[@]}" -type f -exec mv '{}' "${dst}/" \;
}

download_glasses ()
{
    local url="${URL_LIST["glasses"]}"
    local zip="glasses.zip"
    local path="${DATADIR}/celeba_glasses"

    check_dset_exists "${path}"
    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[glasses]}"

    local dset_dir="${DATADIR}/glasses"

    for subdir in {trainA,trainB,testA,testB}
    do
        echo "Restructuring directory: '${subdir}'"
        for splitdir in {1,2}
        do
            # shellcheck disable=SC2178
            local src="${dset_dir}/${subdir}/${splitdir}"
            local dst="${dset_dir}/${subdir}/"

            # shellcheck disable=SC2128
            move_files "${dst}" "${src}"

            # shellcheck disable=SC2128
            exec_or_die rmdir "${src}"
        done
    done

    exec_or_die mv "${dset_dir}" "${path}"
    echo " - Dataset is moved to '${path}'"
}

download_celeba_all ()
{
    # NOTE: This dset is simply restructured male2female
    local url="${URL_LIST["male2female"]}"
    local zip="male2female.zip"
    local path="${DATADIR}/celeba_all"

    check_dset_exists "${path}"
    download_archive "${url}" "${zip}" "${CHECKSUMS[male2female]}"

    exec_or_die unzip "${DATADIR}/${zip}" -d "${path}"

    local unzipped_path="${path}/celeba_male2female"

    exec_or_die mkdir -p "${path}/train" "${path}/val"

    exec_or_die mv "${unzipped_path}/trainA" "${unzipped_path}/trainB" \
        "${path}/train/"

    exec_or_die mv "${unzipped_path}/testA" "${unzipped_path}/testB" \
        "${path}/val"

    exec_or_die rmdir "${unzipped_path}"
    echo " - Dataset is moved to '${path}'"
}

display_hq_resize_warning ()
{
    cat <<'EOF'

NOTE: If you would like to reproduce UVCGANv2 paper results with any
of the high-quality datasets (Celeba-HQ or AFHQ), please resize the
downloaded dataset with `scripts/downsize_right.py` script, like:

python scripts/downsize_right.py SOURCE TARGET -i lanczos -s 256 256

EOF
}

download_celeba_hq ()
{
    local url="${URL_LIST["celeba_hq"]}"
    local zip="celeba_hq.zip"
    local path="${DATADIR}/celeba_hq"

    check_dset_exists "${path}"
    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[celeba_hq]}"

    display_hq_resize_warning
}

download_afhq ()
{
    local url="${URL_LIST["afhq"]}"
    local zip="afhq.zip"
    local path="${DATADIR}/afhq"

    check_dset_exists "${path}"
    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[afhq]}"

    display_hq_resize_warning
}

dataset="${1}"

case "${dataset}" in
    selfie2anime|anime2selfie)
        download_anime2selfie
        ;;
    male2female|celeba_male2female_preproc)
        download_male2female
        ;;
    glasses|eyeglasses|celeba_glasses_preproc)
        download_glasses
        ;;
    celeba_all|celeba_preproc)
        download_celeba_all
        ;;
    celeba_hq|celebahq)
        download_celeba_hq
        ;;
    afhq)
        download_afhq
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage "Unknown dataset '${dataset}'"
esac

