#!/usr/bin/env bash

OUTDIR="${UVCGAN2_OUTDIR:-outdir}"
ZENODO_BASE="https://zenodo.org/record/7826901"

declare -A EXTRACT_PATHS=(
    [afhq_cat2dog]="afhq/cat2dog"
    [afhq_wild2cat]="afhq/wild2cat"
    [afhq_wild2dog]="afhq/wild2dog"
    [afhq_pretrain]="afhq/pretrain"
    [anime2selfie]="anime2selfie/anime2selfie"
    [anime2selfie_pretrain]="anime2selfie/pretrain"
    [celeba_glasses]="celeba/glasses"
    [celeba_male2female]="celeba/male2female"
    [celeba_pretrain]="celeba/pretrain"
    [celeba_hq_male2female]="celeba_hq/male2female"
    [celeba_hq_pretrain]="celeba_hq/pretrain"
)

declare -A CUSTOM_PATHS=(
    [afhq_pretrain]="afhq_resized_lanczos/model_m(simple-autoencoder)_d(None)_g(vit-modnet)_pretrain-uvcgan2"
    [anime2selfie_pretrain]="anime2selfie/model_m(autoencoder)_d(None)_g(vit-modnet)_pretrain-uvcgan2"
    [celeba_pretrain]="celeba/model_m(simple-autoencoder)_d(None)_g(vit-modnet)_pretrain-uvcgan2"
    [celeba_hq_pretrain]="celeba_hq_resized_lanczos/model_m(simple-autoencoder)_d(None)_g(vit-modnet)_pretrain-uvcgan2"
)

# NOTE: regen command
# $ sha256sum *.zip | awk '{printf "[%s]=%s\n", $2, $1}'

declare -A CHECKSUMS=(
    [afhq_cat2dog_full.zip]=cad69806b03ec8d9b3afa9d6ade1796da4042a3c22a111d3ec4c8c803cce3887
    [afhq_cat2dog_only_gen.zip]=d73bbfdf9960905e07c9e8154f7894075244c806b06ea02fd28e1920978aa813
    [afhq_pretrain_full.zip]=e5b363afb0b29253a06d7047f42ed5ede69f5a06a7cf863c15d4322bf33f8df6
    [afhq_pretrain_only_gen.zip]=d057bbc3af917d2571781d2ee2615a3e3badafeade00554dd8b432091493fa82
    [afhq_wild2cat_full.zip]=48ead0b3af8f7c582b3b895d54ceda1f3f0b9e1815f83f0f1c11bff4d028de88
    [afhq_wild2cat_only_gen.zip]=e2f081ebdb49247d5926335bbe749c56608508239ea1f2a790a98fcd0510c6f8
    [afhq_wild2dog_full.zip]=2b07133e6793197fb512f154d6183f5c12cacee48173bb672c8c52dc38b72055
    [afhq_wild2dog_only_gen.zip]=6bdd314e29ba918b43e913d9bffad4fee91f7ec2cb2a4af29c3b9300f8a4e992
    [anime2selfie_anime2selfie_full.zip]=40406a8bfcfe4b3746df11c01e74b58e722f1fbdf0cd61c3f841ef0d9922c40f
    [anime2selfie_anime2selfie_only_gen.zip]=fc1f473f25b252a47f5a20d9549823936fff66eea684e46a51d5e49c76212555
    [anime2selfie_pretrain_full.zip]=1084a3f43a2b993b0f50cacf2a5722c1217e1398abe9f2b9056a2c663aadfc84
    [anime2selfie_pretrain_only_gen.zip]=694e8321aa85fd4b6bf02ea52d0ee002df39682b76a344ab427b0b222bf5e130
    [celeba_glasses_full.zip]=56ff2eebfa8b5760c590fdd7e8e77d986208660a2615fd36d96dd6b5051f0ba9
    [celeba_glasses_only_gen.zip]=c2c8c5eafaabf46d3fea5a01163563ce56b5af4387383fa9bec3db29468d3344
    [celeba_hq_male2female_full.zip]=b86b0e04885f95666e1915a1b18f81dbd5620ceef3ccc365836e9e340229bf5f
    [celeba_hq_male2female_only_gen.zip]=e8bdb39d11d66693a7de556c8cad7b087a06ca6cc9bd0fd8f3f35d4bd7bc207c
    [celeba_hq_pretrain_full.zip]=8d994a8dc479e4d770d3f398d948fc457ecb5275d1cfdf6927a6e690849deed7
    [celeba_hq_pretrain_only_gen.zip]=bbcab4eaade1ac244b40dd0379da4acd401fc974da2d0f190c84c7b53ed92103
    [celeba_male2female_full.zip]=e07f9eeece4b2c19f76150f2fa96eb794adc7c88259945760b0e651a096ee19e
    [celeba_male2female_only_gen.zip]=23c96eddea880ce82f2e49af6a62b78a1dbe598d8adcf0605aefd7af164201b0
    [celeba_pretrain_full.zip]=27a1735860b37d2366d064c19cea31140fffc99fed3419b1bd9a8b1755167d4e
    [celeba_pretrain_only_gen.zip]=81110b96078a176f2676fcb791d6398c79ad5d0e46d1a9be70e4e9cca81979b3
)

die ()
{
    echo "${*}"
    exit 1
}

usage ()
{
    cat <<EOF
USAGE: download_model.sh [-f|--full] [-h|--help] MODEL

where MODEL is one of ${!EXTRACT_PATHS[@]}.

Download and extract a pre-trained model. By default this script will
download translation generators only. If --full is specified, then the entire
training state (state of optimizers, schedulers, etc) of the model will be
downloaded.

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

get_file_name ()
{
    local dataset="${1}"
    local full="${2}"

    if [[ "${full}" == 1 ]]
    then
        echo "${dataset}_full.zip"
    else
        echo "${dataset}_only_gen.zip"
    fi
}

calc_sha256_hash ()
{
    local path="${1}"
    sha256sum "${path}" | cut -d ' ' -f 1 | tr -d '\n'
}

download_zenodo_file ()
{
    local archive="${1}"
    local dest="${2}"

    local url="${ZENODO_BASE}/files/${archive}"

    exec_or_die wget "${url}" --output-document "${dest}"
}

download_archive ()
{
    local archive="${1}"
    local zip_path="${2}"

    if [[ ! -e "${zip_path}" ]]
    then
        exec_or_die mkdir -p "${OUTDIR}"
        download_zenodo_file "${archive}" "${zip_path}"
    fi

    local null_csum="${CHECKSUMS[${archive}]}"

    if [[ -n "${null_csum}" ]]
    then
        # shellcheck disable=SC2155
        local test_csum="$(calc_sha256_hash "${zip_path}")"

        if [[ "${test_csum}" == "${null_csum}" ]]
        then
            echo " - Checksum valid"
        else
            die "Checksum mismatch for '${zip_path}' "\
                "${test_csum} vs ${null_csum}"
        fi
    fi
}

download_and_extract_zip ()
{
    local archive="${1}"
    local zip_path="${2}"
    local extract_path="${3}"

    download_archive  "${archive}" "${zip_path}"
    exec_or_die unzip "${zip_path}" -d "${OUTDIR}"

    echo " - Model is downloaded to: '${zip_path}'"
    echo " - Model is unpacked to: '${extract_path}'"
}

check_model_exists ()
{
    local path="${1}"

    if [[ -e "${path}" ]]
    then

        read -r -p "Model '${path}' exists. Overwrite? [yN]: " ret
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

download_model ()
{
    local model="${1}"
    local full="${2}"

    # shellcheck disable=SC2155
    local archive="$(get_file_name "${model}" "${full}")"

    local zip_path="${OUTDIR}/${archive}"
    local extract_path="${OUTDIR}/${EXTRACT_PATHS[$model]}"

    check_model_exists "${extract_path}"
    download_and_extract_zip "${archive}" "${zip_path}" "${extract_path}"

    local custom_path="${CUSTOM_PATHS[${model}]}"

    if [[ -n "${custom_path}" ]]
    then
        cat<<EOF

NOTE: if you would like to use this pretrained model for transfer training
of the image translation models, move it to:

${OUTDIR}/${custom_path}

EOF
    fi
}

MODEL=
FULL=

while [ $# -gt 0 ]
do
    case "$1" in
        -h|--help|help)
            usage
            ;;
        -f|--full)
            FULL=1
            shift
            ;;
        *)
            [[ -z "${EXTRACT_PATHS[$1]}" ]] && usage "Unknown model $1"
            [[ -n "${MODEL}" ]] && usage "Model is already specified"

            MODEL="${1}"
            shift
            ;;
    esac
done

[[ -z "${MODEL}" ]] && usage
download_model "${MODEL}" "${FULL}"

