#!/bin/sh

# ------------------------------------------------------------------------------
#
# General script for running sim_hessarray simulations for 
#       a HESS 4-telescope system (phase 1 variants), or
#       the HESS-2 5-telescope system in hybrid mode, or
#       a CTA benchmark array (9 telescopes), or
#       a system of 41 HESS-1 telescopes, or
#       the 97 telescope array made up of super-HESS-1 and super-HESS-2 types.
#
# There are two typical operation modes:
#   a) The Corsika data is stored on disk and the run parameters are registered
#      in the database.
#      In this case the run number will be assumed to be the first parameter
#      and the pointing direction and primary id are used from the database.
#   b) The Corsika data comes in standard input, with some environment
#      variables set up by the multipipe_corsika program.
#      In this case the run number, pointing direction and primary id are
#      taken from the CORSIKA_* enviroment variables:
#          CORSIKA_RUN
#          CORSIKA_THETA
#          CORSIKA_PHI
#          CORSIKA_PRIMARY (e.g. '1' for gamma rays)
#
# There are also ways to override settings in either mode through variables
#   offset:       Offset angle to (center of) generated shower direction. [deg]
#   conv_depth:   Atmospheric depth at which telescope directions should converge.
#   transmission: Name of atmospheric transmission table.
#   pixels:       Number of pixels required for trigger (note: '3' means '2.5').
#   threshold:    Pixel threshold in millivolts.
#   zenith_angle: In deg.
#   azimuth_angle:In deg.
#   primary:      Name of primary particle type (�gamma� for gamma rays).
#   nsb:          nighsky background p.e. rate [GHz]
#   reprocessing: Old output files should be regenerated.
#   repro_older_than: If reprocessing=1 then files older than this will be
#       regenerated while for existing newer output files no action is taken.
#
# Defaults are for parallel viewing (conv_depth=0) towards the generated 
# direction (or center of direction cone or range, if applicable) with
# desert haze model (base of boundary layer at HESS site). The NSB rate
# defaults to 100 MHz (i.e. 0.1) for vertical, slightly increasing with
# zenith angle.
#
# Only the hess_bestguess.cfg configuration file will be used. All other
# parameters will be passed on the command line.
#
# ------------------------------------------------------------------------------

export LC_ALL=C

echo "$$ starting"

if [ -z "$cfg" ]; then
   cfg="$(basename $0 | sed 's/^\(.*\)_run.sh$/\1/' | sed 's/^\(.*\).sh$/\1/')"
fi

if [ -z "${SIM_TELARRAY_PATH}" ]; then
   SIM_TELARRAY_RUN_PATH="$(cd $(dirname $0) && pwd -P)"
   if [ "${SIM_TELARRAY_RUN_PATH}" = "." ]; then
      SIM_TELARRAY_RUN_PATH="$(pwd -P)"
   fi
   if [ "$(dirname ${SIM_TELARRAY_RUN_PATH})" = "sim_telarray" ]; then
      SIM_TELARRAY_PATH="$(dirname ${SIM_TELARRAY_RUN_PATH})"
   else
      SIM_TELARRAY_PATH="${SIM_TELARRAY_RUN_PATH}"
   fi
fi

progpath="${SIM_TELARRAY_PATH}"

cd "$progpath" || exit 1

echo "Working directory is `/bin/pwd`"

if [ ! -z "${MCDATA_PATH}" ]; then
   mcdatapath="${MCDATA_PATH}"
else
   mcdatapath="Data"
fi
if [ ! -d "${mcdatapath}" ]; then
   if [ ! -z "${HESSMCDATA}" ]; then
      mcdatapath="${HESSMCDATA}"
   else
      if [ -e "${HOME}/mcdata" ]; then
         mcdatapath="${HOME}/mcdata"
      fi
   fi
fi

echo "Data path is ${mcdatapath}"

if [ -z "$offset" ]; then
   echo "Offset angle is missing"
   exit 1
fi

if [ -z "$conv_depth" ]; then
   conv_depth="0"
fi

if [ -z "$CORSIKA_RUN" ]; then
   runnum="$1"
   shift
else
   runnum="$CORSIKA_RUN"
fi

more_config="$extra_config $*"
echo "$$: Run number: $runnum, conv_depth=$conv_depth, offset=$offset"
echo "$$: Config options: $more_config"

export PATH=$PATH:${SIM_TELARRAY_PATH}/bin:.

if [ -z $zenith_angle ]; then
   zenith_angle="$CORSIKA_THETA"
fi
zenith_angle2="`echo ${zenith_angle}'+'${offset} | bc -l`"
if [ -z $azimuth_angle ]; then
   azimuth_angle="$CORSIKA_PHI"
fi
if [ "$azimuth_angle" = "360" ]; then
   azimuth_angle="0"
fi

plidx="2.68"

if [ -z $primary ]; then
   primary_id="$CORSIKA_PRIMARY"
   primary="unknown"
   case $primary_id in
      1)
         primary="gamma" 
	 plidx="2.50" ;;
      2)
         primary="positron"
	 plidx="3.30" ;;
      3)
         primary="electron"
	 plidx="3.30" ;;
      [56])
         primary="muon" ;;
      14)
         primary="proton" ;;
      402)
         primary="helium" ;;
      1206)
         primary="carbon" ;;
      1407)
         primary="nitrogen" ;;
      1608)
         primary="oxygen" ;;
      2412)
         primary="magnesium" ;;
      2814)
         primary="silicon" ;;
      4020)
         primary="calcium" ;;
      5626)
         primary="iron" ;;
   esac
fi

if [ "$primary" = "unknown" ]; then
   echo "Fatal: Cannot identify primary particle type $primary for run $runnum."
   exit 1
fi

if [ "$primary" != "gamma" ]; then
   if [ "$offset" != "0.0" ]; then
      echo "Fatal: Refusing to run non-zero offset simulations for $primary primaries."
      exit 1
   fi
fi

usecone="unknown"
nonzerocone="0"
if [ ! -z "${CORSIKA_CONE}" ]; then
   usecone=$(echo "${CORSIKA_CONE} > 1.5" | bc)
   nonzerocone=$(echo "${CORSIKA_CONE} > 0.01" | bc)
fi

#fi


cfgnm="$cfg"
case "$cfg" in
   cta) cfgnm="cta1" ;;
   hess_41tel) cfgnm="hess41" ;;
esac

if [ "$conv_depth" = "0" ]; then
   basecfg="${cfgnm}${rotflag}"
   basecfgpath="${cfgnm}"
else
   basecfg="${cfgnm}${rotflag}conv${conv_depth}"
   basecfgpath="${cfgnm}conv"
fi

if [ -z "$CORSIKA_RUN" ]; then
   inputpath="${mcdatapath}/corsika"

   indir=run`printf '%06d' "$runnum"`

   if [ -d "${inputpath}/${indir}/tmp" ]; then
      inputs="`ls ${inputpath}/${indir}/tmp/run*.corsika* 2>/dev/null | tail -1`"
   else
      inputs="`ls ${inputpath}/${indir}/run*.corsika* 2>/dev/null | tail -1`"
   fi

   if [ "x${inputs}" = "x" ]; then
      echo "No input file for run $runnum"
      exit 1
   fi
else
#  Standard input:
   inputs="-"
fi

outputpath="${mcdatapath}/sim_hessarray/${basecfgpath}/${offset}deg"
outprefix="${primary}_${zenith_angle}deg_${azimuth_angle}deg_run${runnum}"
tempprefix="temp_${zenith_angle}deg_${azimuth_angle}deg_run${runnum}"

# If desired we could create the output path as needed:
#if [ ! -d ${outputpath} ]; then
#   mkdir -p ${outputpath}{Data,Log,Histograms}
#fi

if [ "$offset" = "0.0" ]; then
   output_name="${outprefix}_${pixels}_${threshold}_${basecfg}${transtype}${extra_suffix}"
   temp_name="${tempprefix}_${pixels}_${threshold}_${basecfg}${transtype}${extra_suffix}"

else
   output_name="${outprefix}_${pixels}_${threshold}_${basecfg}${transtype}${extra_suffix}_off${offset}"
   temp_name="${tempprefix}_${pixels}_${threshold}_${basecfg}${transtype}${extra_suffix}_off${offset}"
fi

if [ "$primary" = "gamma" -a "$nonzerocone" = "1" ]; then
   output_name="${output_name}_cone${CORSIKA_CONE}"
fi

output_file="${outputpath}/Data/${output_name}.simhess.gz"
hdata_file="${outputpath}/Histograms/${output_name}.hdata.gz"
log_file="${outputpath}/Log/${output_name}.log"
temp_file="${outputpath}/Data/${temp_name}.dat"

if [ -f "${output_file}" ]; then
   if [ "$reprocessing" = "1" ]; then
      if [ -f "${output_file}.old" ]; then
         echo "Reprocessing requested but ${output_file}.old already exists. Nothing done."
	 exit 1;
      else
         touch --date "$repro_older_than" "${output_file}.test"
	 if [ "${output_file}" -nt "${output_file}.test" ]; then
	    echo "Reprocessing requested but ${output_file} is a new file. Nothing done."
	    rm "${output_file}.test"
	    exit 1;
	 fi
	 rm "${output_file}.test"
         mv "${output_file}" "${output_file}.old" || exit 1
	 echo "Reprocessing requested: file ${output_file} renamed to ${output_file}.old."
	 mv -f ${hdata_file} ${hdata_file}.old
	 mv -f ${log_file} ${log_file}.old
	 mv -f ${log_file}.gz ${log_file}.gz.old
      fi
   fi
fi

if [ -f "${output_file}" ]; then
   echo "Output file ${output_file} already exists. Nothing done."
   exit 1;
fi

defs="${defs} -DCTA_ULTRA3"
extraopt="-Icfg/CTA"
maxtel=100
iobufmx=2000000000

# Some setups are available for different altitudes
# (note: must match the atmospheric transmission file).
#if [ ! -z "${altitude}" ]; then
#   extraopt="${extraopt} -C Altitude=${altitude}"
#fi

# If the NSB is not used as given in the config file, set it here:
if [ ! -z "${nsb}" ]; then
   extraopt="${extraopt} -C nightsky_background=all:${nsb}"
fi

# Show what is supposed to be done:

echo "$$: Starting: " ./bin/sim_hessarray -c "${cfgfile}" \
        ${defs} ${extraopt} \
	-C "iobuf_maximum=${iobufmx}" \
	-C "maximum_telescopes=${maxtel}" \
	-C "atmospheric_transmission=$transmission" \
        -C "convergent_depth=${conv_depth}" \
	-C "telescope_theta=${zenith_angle2}" \
	-C "telescope_phi=${azimuth_angle}" \
	-C "power_law=${plidx}" \
	-C "histogram_file=${hdata_file}" \
	-C "output_file=${output_file}" \
	-C "random_state=auto" \
	$more_config \
	-C "show=all" \
	${inputs} "| gzip > ${log_file}.gz"

if [ "$testonly" = 1 ]; then
   echo "This was just a script test and nothing actually run."
   exit 1;
fi

if [ ! -x ./sim_hessarray ]; then
   echo "Cannot run ./sim_hessarray: no such file or not executable"
   exit 1
fi

# Now start to do the real work:

./bin/sim_hessarray -c "${cfgfile}" \
        ${defs} ${extraopt} \
	-C "iobuf_maximum=${iobufmx}" \
	-C "maximum_telescopes=${maxtel}" \
	-C "atmospheric_transmission=$transmission" \
        -C "convergent_depth=${conv_depth}" \
	-C "telescope_theta=${zenith_angle2}" \
	-C "telescope_phi=${azimuth_angle}" \
	-C "power_law=${plidx}" \
	-C "histogram_file=${hdata_file}" \
	-C "output_file=${output_file}" \
	-C "random_state=auto" \
	$more_config \
	-C "show=all" \
	${inputs} \
	 2>&1 | gzip > "${log_file}.gz"

