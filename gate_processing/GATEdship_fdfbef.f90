#define FILENAME_LENGHT 132
module GATEdship_mod

  type :: datetime
     integer :: year
     integer :: month
     integer :: day
     integer :: hour
     integer :: minute
     integer :: second
  end type datetime

  type :: position
     integer :: deg
     integer :: min
     integer :: sec
  end type position

  type :: GATE_dship_type
     ! do not change the sequence within this type
     integer :: DATE
     integer :: TIME
     real    :: LATITUDE
     real    :: LONGITUDE
     real    :: SHIP_SPEED
     integer :: CNTS_SHIP_SPEED
     real    :: SHIP_HEADING
     integer :: CNTS_SHIP_HEAD
     real    :: INC_SOL_RAD
     integer :: CNTS_INC_SOL
     real    :: REFL_SOL_RAD
     integer :: CNTS_REFL_SOLAR
     real    :: NET_RADIATION
     integer :: CNTS_NET_RAD
     real    :: KOLLS_PRESSURE
     integer :: CNTS_KOLLS_PRES
     real    :: ROSE_PRESSURE
     integer :: CNTS_ROSE_PRES
     real    :: SEA_SURF_TEMP
     integer :: CNTS_SURF_TEMP
     real    :: DRY_BULB_TEMP
     integer :: CNTS_DRY_BULB
     real    :: WET_BULB_TEMP1
     integer :: CNTS_WET_BULB1
     real    :: WET_BULB_TEMP2
     integer :: CNTS_WET_BULB2
     real    :: SPEC_HUMIDITY1
     real    :: SPEC_HUMIDITY2
     real    :: DEW_PT_TEMP1
     real    :: DEW_PT_TEMP2
     real    :: WIND_SPEED_BOOM
     integer :: CNTS_W_S_BOOM
     real    :: WIND_SPEED_MAST
     integer :: CNTS_W_S_MAST
     real    :: WIND_DIR_BOOM
     integer :: CNTS_W_D_BOOM
     real    :: WIND_DIR_MAST
     integer :: CNTS_W_D_MAST
     real    :: WIND_U_COM_BOOM
     real    :: WIND_V_COM_BOOM
     real    :: WIND_U_COM_MAST
     real    :: WIND_V_COM_MAST
  end type GATE_dship_type

  type :: GATE_metadata_type

     character(len=32) :: shipname
     type (datetime)   :: measurement_time_start
     real              :: interval
     character         :: interval_unit

     character(len=4)  :: pressure_unit='mbar'
     character(len=3)  :: radiation_unit='W/m**2'
     character(len=4)  :: temperature_unit='degC'
     character(len=4)  :: humidity_unit='g/kg'
     character(len=3)  :: wind_unit='m/s' 

  end type GATE_metadata_type

  public :: GATE_dship_type, GATE_metadate_type

end module GATEdship_mod

! ----------------------

program GATEdship

  ! ASCII Input
  !
  ! /work/mh0010/GATE/NOAA_data/Sorted_files/
  !     3.00.02.104-3.31.02.101_19740601-19740930/
  !     02_Surface_meteorological_data_different_averages_of_all_meteorological_variables_(ships)

  use, intrinsic :: iso_fortran_env, only : iostat_end

  implicit none

  integer :: i
  integer :: ierror = 0

  character(len=FILENAME_LENGHT)   :: infile
  character(len=FILENAME_LENGHT)   :: line

  call getarg( 1, line )
  infile = trim(line)

  open (unit=10, file=infile, status='old', form='formatted', action='read')
  read( 10, '(A80)', iostat=ierror ) line(1:80)

  select case (line(1:1))

  case ( "0" )

     rewind(10)
     read( 10, '(A)', iostat=ierror ) line(1:80)
     read( 10, '(A)', iostat=ierror ) line(1:80)
     write ( * , * )
     write ( * , * ) 'Found tape header record type 0 in'
     write ( * , * ) trim(infile)
     write ( * , * ) 'Type of computer used : ', line(26:37)
     read( 10, '(A)', iostat=ierror ) line(1:80)
     write ( * , * ) 'Translation table:'
     write ( * , * ) line(2:54)
     close (unit=10)

  case ( "1" )

     rewind(10)
     do i = 1, 8
        read( 10, '(A)', iostat=ierror ) line(1:80)
     enddo
     close (unit=10)
 
     if ( line(2:57) /= "(I1,I4,I10,I5,46X,6(I6,I7,2F7.3,F6.1,I6,F6.0,I6,F6.1,I6," ) then
        write ( * , * )
        write ( * , * ) trim(infile), ':'
        write ( * , * ) ' - wrong format description'
        stop
     end if

     write ( * , * )
     write ( * , * ) 'now processing ', trim(infile)
     call convert_data ( infile )

  case default
     close (unit=10)
     write ( * , * )
     write ( * , * ) trim(infile), ' does not contain data'
     stop

  end select

end program GATEdship

! ----------------------

subroutine convert_data (infile)

  use, intrinsic :: iso_fortran_env, only : iostat_end
  use GATEdship_mod

  implicit none

  character(len=FILENAME_LENGHT), intent(in) :: infile

  ! 4 ints plus 6 * 46 measurements per full data line (see format string)

  integer :: type_id          ! file header record type indicator
  integer :: records_in_line  ! number of of records contained in one full data line
  integer :: records_handled  ! number of records allready stored away
  integer :: line_number      ! of one line containing full format string

  type (GATE_dship_type) :: shipdata(6)

  ! array for keeping the whole profile
  type (GATE_dship_type), pointer :: dshipdata(:)
  type (GATE_dship_type), allocatable, target :: dshipcurrent(:)
  type (GATE_dship_type), allocatable, target :: dshiptempory(:)
  ! measurement and field metadata
  type (GATE_metadata_type) :: metadata

  integer :: ierror = 0
  integer :: no_records
  integer :: no_of_measurement

  integer :: i
  integer :: dshipsize, dshipsize_inc

  character(len=24*80) :: line

  dshipsize_inc = 48
  dshipsize     = dshipsize_inc

  allocate (dshipcurrent(dshipsize))
  dshipdata => dshipcurrent

  open (unit=10, file=trim(infile), status='old', form='formatted', action='read')

  ! file section 1, summary of measurement

  do i = 1, 24

     read( 10, 100, iostat=ierror ) line(1:80)

     select case (ierror)

     case ( 0 )

        if ( i == 2 ) then
           metadata%shipname = line(16:39)
        end if

        if ( i == 7 ) then
           read(line(17:25),'(F9.1)') metadata%interval
           read(line(26:26),'(A1)')   metadata%interval_unit
           ! write ( * , * ) metadata%interval,metadata%interval_unit 
        end if

        if ( i == 4 ) then
           read(line(2:15),'(i4,5i2)')                  &
                metadata%measurement_time_start%year,   &
                metadata%measurement_time_start%month,  &
                metadata%measurement_time_start%day,    &
                metadata%measurement_time_start%hour,   &
                metadata%measurement_time_start%minute, &
                metadata%measurement_time_start%second
        end if

     case ( iostat_end )
        write ( * , * ) 'Unexpectedly reached end of file in section 1!'
        exit
     case default
        write ( * , * ) 'Unexpected error when reading section 1!'
        exit
     end select
  end do

  ! file section 2, metadata of sampled variables

  do i = 1, 96
     read( 10, 100, iostat=ierror ) line(1:80)
     select case (ierror)
     case ( 0 )
        continue
     case ( iostat_end )
        write ( * , * ) 'Unexpectedly reached end of file in section 2!'
        exit
     case default
        write ( * , * ) 'Unexpected error when reading section 2!'
        exit
     end select
  end do

  ! file section 3

  no_of_measurement = 0

  do while ( ierror == 0 )

     ! Read data records, 24 lines for 1 record.

     do i = 1, 24
        read( 10, 100, iostat=ierror ) line((i-1)*80+1:i*80)
        select case (ierror)
        case ( 0 )
           continue
        case ( iostat_end )
           write ( * , * ) 'EOF reached after trying record ', no_of_measurement
           exit
        case default
           write ( * , * ) 'Unexpected error when reading section 3!'
           exit
        end select
     end do

     if ( ierror == 0 ) then

        read(line, 110, iostat=ierror ) type_id, records_in_line, records_handled, line_number, shipdata(1:6)

        if ( ierror /= 0 ) write ( * , * ) 'format error in ', trim(line)
        if ( no_of_measurement >= dshipsize ) then
           allocate (dshiptempory(dshipsize))
           dshiptempory = dshipdata
           deallocate(dshipcurrent)
           dshipsize=dshipsize+dshipsize_inc
           allocate(dshipcurrent(dshipsize))
           dshipcurrent(1:dshipsize-dshipsize_inc) = dshiptempory
           deallocate(dshiptempory)
           dshipdata => dshipcurrent
        end if

        if ( no_of_measurement /= records_handled ) then
           write ( * , * ) 'Inconsistency in number of records.'
           stop 'Stopped'
        end if
        
        do i = 1, records_in_line
           no_of_measurement = no_of_measurement + 1
           dshipdata(no_of_measurement) = shipdata(i)
           ! write ( * , * ) ' measurement ', i, no_of_measurement, shipdata(i)%latitude, shipdata(i)%longitude
         end do

     end if

  end do ! while-loop

  close (10)

  write ( * , * ) 'Processed ', no_of_measurement, 'data records.'

  call write_netcdf ( infile, no_of_measurement, dshipdata, metadata )

  deallocate ( dshipcurrent )

110 format(I1,I4,I10,I5,46X,6(I6,I7,2F7.3,F6.1,I6,F6.0,I6,F6.1,I6, &
           12X,4(F6.1,I6),2(F6.2,I6),12X,2(F6.2,I6),4F6.2,         &
           2(F6.1,I6),2(F6.0,I6),4F6.1,30X))
100 format(a80)

end subroutine convert_data

! ----------------------

subroutine write_netcdf ( infile, no_of_measurements, dshipdata, metadata )

  use GATEdship_mod
  implicit none

  include 'netcdf.inc'

  character(len=FILENAME_LENGHT), intent(in) :: infile
  integer,                        intent(in) :: no_of_measurements
  type (GATE_dship_type),         intent(in) :: dshipdata(no_of_measurements)
  type (GATE_metadata_type),      intent(in) :: metadata

  character(len=FILENAME_LENGHT) :: outfile

  integer, parameter :: ndims = 2
  integer :: ncid
  integer :: dimids(ndims)
  integer :: start(ndims)
  integer :: edge(ndims)

  integer :: measurement_time_id
  integer :: measurement_id, timer_id
  integer :: lat_id, lon_id
  integer :: rsds_id
  integer :: rlus_id 
  integer :: nrad_id
  integer :: wind_speed_boom_id, wind_dir_boom_id
  integer :: wind_speed_mast_id, wind_dir_mast_id
  integer :: v_boom_id,  v_mast_id
  integer :: u_boom_id,  u_mast_id

  real    :: time(no_of_measurements)

  character(len=33) :: seconds_since

  integer :: i

  ! some preparation for writing global attributes

  write ( seconds_since , '(A14,I4,A1,2(I2.2,A1),2(I2.2,A1),I2.2)' ) &
       & 'seconds since ',                        &
       & metadata%measurement_time_start%year,   '-',  &
       & metadata%measurement_time_start%month,  '-',  &
       & metadata%measurement_time_start%day,    ' ',  &
       & metadata%measurement_time_start%hour,   ':',  &
       & metadata%measurement_time_start%minute, ':',  &
       & metadata%measurement_time_start%second

  ! start writing

  write ( * , * ) trim(infile), ':'
  write ( outfile, '(A,A3)' ) trim(infile), '.nc'

  call handle_err(nf_create( outfile, NF_CLOBBER, ncid))

  call handle_err(nf_def_dim(ncid, 'measurement', 1, measurement_id))
  call handle_err(nf_def_dim(ncid, 'time', NF_UNLIMITED, timer_id))

  dimids(1) = measurement_id
  dimids(2) = timer_id

  start(:) = 1
  edge(2)  = no_of_measurements
  edge(1)  = 1

  call handle_err(nf_def_var(ncid, "time", NF_FLOAT, 1, dimids(2), measurement_time_id))

  call handle_err(nf_def_var(ncid, "lat",         NF_FLOAT, ndims, dimids, lat_id))
  call handle_err(nf_def_var(ncid, "lon",         NF_FLOAT, ndims, dimids, lon_id))
  call handle_err(nf_def_var(ncid, "rsds",        NF_FLOAT, ndims, dimids, rsds_id))
  call handle_err(nf_def_var(ncid, "rlus",        NF_FLOAT, ndims, dimids, rlus_id))
  call handle_err(nf_def_var(ncid, "nrad",        NF_FLOAT, ndims, dimids, nrad_id))
  call handle_err(nf_def_var(ncid, "u_boom",      NF_FLOAT, ndims, dimids, u_boom_id))
  call handle_err(nf_def_var(ncid, "u_mast",      NF_FLOAT, ndims, dimids, u_mast_id))
  call handle_err(nf_def_var(ncid, "v_boom",      NF_FLOAT, ndims, dimids, v_boom_id))
  call handle_err(nf_def_var(ncid, "v_mast",      NF_FLOAT, ndims, dimids, v_mast_id))

  call handle_err(nf_put_att_text(ncid, measurement_time_id, 'units', len(seconds_since), seconds_since))
  call handle_err(nf_put_att_text(ncid, measurement_time_id, "calendar", 19, "proleptic_gregorian"))

  call handle_err(nf_put_att_text(ncid, lat_id, "standard_name", 8, "latitude"))
  call handle_err(nf_put_att_text(ncid, lat_id, "units", 13, "degrees_north"))
  call handle_err(nf_put_att_text(ncid, lon_id, "standard_name", 9, "longitude"))
  call handle_err(nf_put_att_text(ncid, lon_id, "units", 12, "degrees_east"))

  call handle_err(nf_put_att_text(ncid, rsds_id, "standard_name", 24, "incoming_solar_radiation"))
  call handle_err(nf_put_att_text(ncid, rsds_id, "units", len(metadata%radiation_unit), metadata%radiation_unit))
  call handle_err(nf_put_att_real(ncid, rsds_id, "_FillValue", NF_REAL, 1, 9.9))

  call handle_err(nf_put_att_text(ncid, rlus_id, "standard_name", 24, "reflective_solar_radiation"))
  call handle_err(nf_put_att_text(ncid, rlus_id, "units", len(metadata%radiation_unit), metadata%radiation_unit))
  call handle_err(nf_put_att_real(ncid, rlus_id, "_FillValue", NF_REAL, 1, 9.9))

  call handle_err(nf_put_att_text(ncid, nrad_id, "standard_name", 17, "net_radiation"))
  call handle_err(nf_put_att_text(ncid, nrad_id, "units", len(metadata%radiation_unit), metadata%radiation_unit))
  call handle_err(nf_put_att_real(ncid, nrad_id, "_FillValue", NF_REAL, 1, 99.99))

  call handle_err(nf_put_att_text(ncid, u_boom_id, "standard_name", 18, "eastward_wind_boom"))
  call handle_err(nf_put_att_text(ncid, u_boom_id, "units", len(metadata%wind_unit), metadata%wind_unit))
  call handle_err(nf_put_att_real(ncid, u_boom_id, "_FillValue", NF_REAL, 1, 999.9))

  call handle_err(nf_put_att_text(ncid, u_mast_id, "standard_name", 18, "eastward_wind_mast"))
  call handle_err(nf_put_att_text(ncid, u_mast_id, "units", len(metadata%wind_unit), metadata%wind_unit))
  call handle_err(nf_put_att_real(ncid, u_mast_id, "_FillValue", NF_REAL, 1, 99.9))

  call handle_err(nf_put_att_text(ncid, v_boom_id, "standard_name", 19, "northward_wind_boom"))
  call handle_err(nf_put_att_text(ncid, v_boom_id, "units", len(metadata%wind_unit), metadata%wind_unit))
  call handle_err(nf_put_att_real(ncid, v_boom_id, "_FillValue", NF_REAL, 1, 999.9))

  call handle_err(nf_put_att_text(ncid, v_mast_id, "standard_name", 19, "northward_wind_mast"))
  call handle_err(nf_put_att_text(ncid, v_mast_id, "units", len(metadata%wind_unit), metadata%wind_unit))
  call handle_err(nf_put_att_real(ncid, v_mast_id, "_FillValue", NF_REAL, 1, 99.9))

  call handle_err(nf_put_att_text(ncid, NF_GLOBAL, "shipname", len(trim(adjustl(metadata%shipname))), &
       trim(adjustl(metadata%shipname))))

  call handle_err(nf_put_att_real(ncid, NF_GLOBAL, "Average_interval", NF_REAL, 1, metadata%interval))

  call handle_err(nf_enddef (ncid))

  select case (metadata%interval_unit)
  case ("M")
     ! time interval in seconds (!!!)
     time(1) = (-1) *  metadata%interval / 2.0
     do i = 2, no_of_measurements
       time(i) = time(i-1) + metadata%interval
     enddo
  case default
     write ( * , * ) "Unsupported units for time interval"
     stop 'Stopped'
  end select

  write ( * , * ) " - ", trim(adjustl(metadata%shipname)), ', sample interval ', metadata%interval

  call handle_err(nf_put_vara(ncid, measurement_time_id, start(2), edge(2), time))

  call handle_err(nf_put_vara(ncid, lat_id,         start, edge, dshipdata(1:no_of_measurements)%latitude))
  call handle_err(nf_put_vara(ncid, lon_id,         start, edge, dshipdata(1:no_of_measurements)%longitude))
  call handle_err(nf_put_vara(ncid, rsds_id,        start, edge, dshipdata(1:no_of_measurements)%inc_sol_rad))
  call handle_err(nf_put_vara(ncid, rlus_id,        start, edge, dshipdata(1:no_of_measurements)%refl_sol_rad))
  call handle_err(nf_put_vara(ncid, nrad_id,        start, edge, dshipdata(1:no_of_measurements)%net_radiation))
  call handle_err(nf_put_vara(ncid, u_boom_id,      start, edge, dshipdata(1:no_of_measurements)%wind_u_com_boom))
  call handle_err(nf_put_vara(ncid, u_mast_id,      start, edge, dshipdata(1:no_of_measurements)%wind_u_com_mast))
  call handle_err(nf_put_vara(ncid, v_boom_id,      start, edge, dshipdata(1:no_of_measurements)%wind_v_com_boom))
  call handle_err(nf_put_vara(ncid, v_mast_id,      start, edge, dshipdata(1:no_of_measurements)%wind_v_com_mast))

  call handle_err(nf_close(ncid))

end subroutine write_netcdf

! ----------------------

subroutine handle_err(status)
  implicit none
  include 'netcdf.inc'

  integer, intent(in) :: status

  if (status .ne. nf_noerr) then
     print *, nf_strerror(status)
     stop 'Stopped'
  endif
end subroutine handle_err
