#define FILENAME_LENGHT 132
module GATEradiosonde_mod

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

  type :: GATE_radiosonde_type
     ! do not change the sequence within this type
     real :: TIME_AFT_LAUNCH
     real :: PRESSURE
     real :: ALTITUDE
     real :: TEMPERATURE
     real :: TEMP_ERROR
     real :: SPECIFIC_HUMDTY
     real :: HUMDTY_ERROR
     real :: WIND_VEL_U_COMP
     real :: U_COMP_ERROR
     real :: WIND_VEL_V_COMP
     real :: V_COMP_ERROR
  end type GATE_radiosonde_type

  type :: GATE_metadata_type

     character(len=32) :: platform
     integer           :: flight_id
     type (datetime)   :: launch_time_start
     type (datetime)   :: launch_time_end
     type (position)   :: launch_lat_start      
     type (position)   :: launch_lon_start     
     type (position)   :: launch_lat_end      
     type (position)   :: launch_lon_end     
     real              :: p_start
     real              :: p_end   
     real              :: height_max

     character(len=4)  :: p_units='mbar'
     character(len=3)  :: alt_units='gpm'
     character(len=4)  :: t_units='degC'
     character(len=4)  :: sh_units='g/kg'
     character(len=3)  :: wind_units='m/s' 

  end type GATE_metadata_type

  public :: GATE_radiosonde_type, GATE_metadate_type

end module GATEradiosonde_mod

! ----------------------

program GATEradiosonde

  ! ASCII input
  ! 3.00.02.104-3.31.02.101_19740601-19740930
  ! 3.30.21.101_-_3.31.02.101_R_6-8,1,1_19740601-19740930-003 -
  ! 3.30.21.101_-_3.31.02.101_R_6-8,1,1_19740601-19740930-479

  use, intrinsic :: iso_fortran_env, only : iostat_end

  implicit none

  integer :: ierror = 0

  character(len=FILENAME_LENGHT)   :: infile
  character(len=FILENAME_LENGHT)   :: line

  call getarg(1, line)
  infile = TRIM(line)

  write ( * , * ) "Handling ", infile

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
     read( 10, '(A)', iostat=ierror ) line(1:80)
     close (unit=10)
     if ( VERIFY("RADIOSONDE",line(16:39)) /= 0 ) then
       write ( * , * )
       write ( * , * ) trim(infile), ' does not contain radiosonde data',line(16:39)
       stop
     end if

     write ( * , * )
     write ( * , * ) 'now processing ', trim(infile)
     call convert_data ( infile )

  case default
     close (unit=10)
     write ( * , * )
     write ( * , * ) trim(infile) , ' does not contain data'
     stop
 
  end select

end program GATEradiosonde

! ----------------------

subroutine convert_data (infile)

  use, intrinsic :: iso_fortran_env, only : iostat_end
  use GATEradiosonde_mod

  implicit none

  character(len=FILENAME_LENGHT), intent(in) :: infile

  ! 5 ints plus 46 levels per data line (record)
  integer :: id1, id2, id3, id4, LAUNCH_TIME
  type (GATE_radiosonde_type) :: tmpdata(46)

  ! array for keeping the whole profile
  type (GATE_radiosonde_type), allocatable :: radiosondedata(:)
  ! launch and field metadata
  type (GATE_metadata_type) :: metadata

  integer :: ierror = 0
  integer :: no_records
  integer :: no_of_levels

  integer :: level
  integer :: i

  character(len=24*80) :: line

  open (unit=10, file=trim(infile), status='old', form='formatted', action='read')

  ! file section 1, summary of launch

  do i = 1, 24

     read( 10, 100, iostat=ierror ) line(1:80)

     select case (ierror)

     case ( 0 )

        if ( i == 3 ) then
           metadata%platform = line(16:39)
        end if

        if ( i == 4 ) then
           read(line(2:15),'(i4,5i2)')             &
                metadata%launch_time_start%year,   &
                metadata%launch_time_start%month,  &
                metadata%launch_time_start%day,    &
                metadata%launch_time_start%hour,   &
                metadata%launch_time_start%minute, &
                metadata%launch_time_start%second
           read(line(19:33),'(i3,2i2,i4,2i2)') &
                metadata%launch_lat_start%deg, &
                metadata%launch_lat_start%min, &
                metadata%launch_lat_start%sec, &
                metadata%launch_lon_start%deg, &
                metadata%launch_lon_start%min, &
                metadata%launch_lon_start%sec
        end if

        if ( i == 5 ) then
           read(line(2:15),'(i4,5i2)')           &
                metadata%launch_time_end%year,   &
                metadata%launch_time_end%month,  &
                metadata%launch_time_end%day,    &
                metadata%launch_time_end%hour,   &
                metadata%launch_time_end%minute, &
                metadata%launch_time_end%second
           read(line(19:33),'(i3,2i2,i4,2i2)')   &
                metadata%launch_lat_end%deg, &
                metadata%launch_lat_end%min, &
                metadata%launch_lat_end%sec, &
                metadata%launch_lon_end%deg, &
                metadata%launch_lon_end%min, &
                metadata%launch_lon_end%sec
        end if

        if ( i == 17 ) read(line(1:80),'(62X,i6)') no_of_levels

     case ( iostat_end )
        write ( * , * ) 'Unexpectedly reached end of file in section 1!'
        exit
     case default
        write ( * , * ) 'Unexpected error when reading section 1!'
        exit
     end select
  end do

  write ( * , * ) trim(infile), ' contains ', no_of_levels, ' levels.'

  allocate ( radiosondedata(no_of_levels) )

  ! file section 2, metadata of sampled variables

  do i = 1, 24
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

  level = 1
  no_records = 1

  do while ( ierror == 0 )

     ! Read data records, 24 lines for 1 record.

     do i = 1, 24
        read( 10, 100, iostat=ierror ) line((i-1)*80+1:i*80)
        select case (ierror)
        case ( 0 )
           continue
        case ( iostat_end )
           write ( * , * ) 'EOF reached after trying record ', no_records
           no_records = no_records - 1
           exit
        case default
           write ( * , * ) 'Unexpected error when reading section 3!'
           exit
        end select
     end do

     if ( ierror == 0 ) then

        read(line, 110, iostat=ierror ) id1, id2, id3, id4, launch_time, tmpdata(1:46)
        if ( ierror /= 0 ) write ( * , * ) 'format error in ', trim(line)
        do i = 1, 46
           if ( level <= no_of_levels ) then
              radiosondedata(level) = tmpdata(i)
              ! write ( * , * ) launch_time, level, radiosondedata(level)%altitude
           else
              exit
           end if
           level = level + 1
        end do

        no_records = no_records + 1

     end if

  end do ! while-loop

  close (10)

  write ( * , * ) 'Processed ', no_records, 'data records for ', level-1, 'levels.'

  call write_netcdf ( infile, level-1, radiosondedata, metadata )

  deallocate ( radiosondedata )

110 format(I1,I4,I10,I5,I6,54X,46(F4.0,F5.1,F5.0,F4.1,F2.1,F4.2,F2.2,2(F4.1,F3.1)))
100 format(a)

end subroutine convert_data

! ----------------------

subroutine write_netcdf ( infile, no_of_levels, radiosondedata, metadata )

  use GATEradiosonde_mod
  implicit none

  include 'netcdf.inc'

  character(len=FILENAME_LENGHT), intent(in) :: infile
  integer,                     intent(in) :: no_of_levels
  type (GATE_radiosonde_type), intent(in) :: radiosondedata(no_of_levels)
  type (GATE_metadata_type),   intent(in) :: metadata

  character(len=FILENAME_LENGHT) :: outfile
  character(len=18)  :: position_str

  integer, parameter :: ndims = 2
  integer :: ncid
  integer :: dimids(ndims)
  integer :: start(ndims)
  integer :: edge(ndims)

  integer :: launch_time_id
  integer :: level_id, timer_id
  integer :: flight_time_id
  integer :: p_id
  integer :: alt_id
  integer :: t_id, terr_id
  integer :: sh_id, sherr_id
  integer :: u_id,  uerr_id
  integer :: v_id,  verr_id

  real    :: time(1)
  real    :: position_start_lon, position_start_lat
  real    :: position_end_lon,   position_end_lat

  character(len=33) :: seconds_since

  integer :: i

  ! some preparation for writing global attributes

  write ( seconds_since , '(A14,I4,A1,2(I2.2,A1),2(I2.2,A1),I2.2)' ) &
       & 'seconds since ',                        &
       & metadata%launch_time_start%year,   '-',  &
       & metadata%launch_time_start%month,  '-',  &
       & metadata%launch_time_start%day,    ' ',  &
       & metadata%launch_time_start%hour,   ':',  &
       & metadata%launch_time_start%minute, ':',  &
       & metadata%launch_time_start%second

  position_start_lon = metadata%launch_lon_start%deg +      &
       &             ( metadata%launch_lon_start%min * 60 + &
       &               metadata%launch_lon_start%sec ) / 3600.0

  position_start_lat = metadata%launch_lat_start%deg +      &
       &             ( metadata%launch_lat_start%min * 60 + &
       &               metadata%launch_lat_start%sec ) / 3600.0

  position_end_lon = metadata%launch_lon_end%deg +      &
       &           ( metadata%launch_lon_end%min * 60 + &
       &             metadata%launch_lon_end%sec ) / 3600.0

  position_end_lat = metadata%launch_lat_end%deg +      &
       &           ( metadata%launch_lat_end%min * 60 + &
       &             metadata%launch_lat_end%sec ) / 3600.0

  ! start writing

  write ( * , * ) ' writing ', trim(infile)
  write ( outfile, '(A,A3)' ) trim(infile), '.nc'

  call handle_err(nf_create( outfile, NF_CLOBBER, ncid))

  call handle_err(nf_def_dim(ncid, 'level', no_of_levels, level_id))
  call handle_err(nf_def_dim(ncid, 'time', NF_UNLIMITED, timer_id))

  dimids(1) = level_id
  dimids(2) = timer_id

  start(:) = 1
  edge(1)  = no_of_levels
  edge(2)  = 1

  call handle_err(nf_def_var(ncid, "time", NF_FLOAT, 1, dimids(2), launch_time_id))

  call handle_err(nf_def_var(ncid, "flight_time", NF_FLOAT, ndims, dimids, flight_time_id))
  call handle_err(nf_def_var(ncid, 'plev',        NF_FLOAT, ndims, dimids, p_id))
  call handle_err(nf_def_var(ncid, 'alt',         NF_FLOAT, ndims, dimids, alt_id))
  call handle_err(nf_def_var(ncid, "ta",          NF_FLOAT, ndims, dimids, t_id))
  call handle_err(nf_def_var(ncid, "ta_err",      NF_FLOAT, ndims, dimids, terr_id))
  call handle_err(nf_def_var(ncid, "hus",         NF_FLOAT, ndims, dimids, sh_id))
  call handle_err(nf_def_var(ncid, "hus_err",     NF_FLOAT, ndims, dimids, sherr_id))
  call handle_err(nf_def_var(ncid, 'ua',          NF_FLOAT, ndims, dimids, u_id))
  call handle_err(nf_def_var(ncid, "ua_err",      NF_FLOAT, ndims, dimids, uerr_id))
  call handle_err(nf_def_var(ncid, 'va',          NF_FLOAT, ndims, dimids, v_id))
  call handle_err(nf_def_var(ncid, "va_err",      NF_FLOAT, ndims, dimids, verr_id))

  call handle_err(nf_put_att_text(ncid, launch_time_id, 'units', len(seconds_since), seconds_since))
  call handle_err(nf_put_att_text(ncid, flight_time_id, 'units', len(seconds_since), seconds_since))

  call handle_err(nf_put_att_text(ncid, p_id, "standard_name", 12, "air_pressure"))
  call handle_err(nf_put_att_text(ncid, p_id, "units", len(metadata%p_units), metadata%p_units))
  call handle_err(nf_put_att_real(ncid, p_id, "_FillValue", NF_REAL, 1, 9999.9))
  call handle_err(nf_put_att_real(ncid, p_id, "missing_value", NF_REAL, 1, 9999.9))

  call handle_err(nf_put_att_text(ncid, alt_id, "standard_name", 19, "geopotential_height"))
  call handle_err(nf_put_att_text(ncid, alt_id, "positive", 2, "up"))
  call handle_err(nf_put_att_text(ncid, alt_id, "units", len(metadata%alt_units), metadata%alt_units))
  call handle_err(nf_put_att_real(ncid, alt_id, "_FillValue", NF_REAL, 1, 99999.0))
  call handle_err(nf_put_att_real(ncid, alt_id, "missing_value", NF_REAL, 1, 99999.0))

  call handle_err(nf_put_att_text(ncid, t_id, "standard_name", 11, "temperature"))
  call handle_err(nf_put_att_text(ncid, t_id, "units", len(metadata%t_units), metadata%t_units))
  call handle_err(nf_put_att_real(ncid, t_id, "_FillValue", NF_REAL, 1, 99.9))
  call handle_err(nf_put_att_real(ncid, t_id, "missing_value", NF_REAL, 1, 99.9))

  call handle_err(nf_put_att_text(ncid, terr_id, "standard_name", 17, "temperature_error"))
  call handle_err(nf_put_att_text(ncid, terr_id, "units", len(metadata%t_units), metadata%t_units))
  call handle_err(nf_put_att_real(ncid, terr_id, "_FillValue", NF_REAL, 1, 9.9))
  call handle_err(nf_put_att_real(ncid, terr_id, "missing_value", NF_REAL, 1, 9.9))

  call handle_err(nf_put_att_text(ncid, sh_id, "standard_name", 17, "specific_humidity"))
  call handle_err(nf_put_att_text(ncid, sh_id, "units", len(metadata%sh_units), metadata%sh_units))
  call handle_err(nf_put_att_real(ncid, sh_id, "_FillValue", NF_REAL, 1, 99.99))
  call handle_err(nf_put_att_real(ncid, sh_id, "missing_value", NF_REAL, 1, 99.99))

  call handle_err(nf_put_att_text(ncid, sherr_id, "standard_name", 23, "specific_humidity_error"))
  call handle_err(nf_put_att_text(ncid, sherr_id, "units", len(metadata%sh_units), metadata%sh_units))
  call handle_err(nf_put_att(ncid, sherr_id, "_FillValue", NF_REAL, 1, 0.99))
  call handle_err(nf_put_att(ncid, sherr_id, "missing_value", NF_REAL, 1, 0.99))

  call handle_err(nf_put_att_text(ncid, u_id, "standard_name", 13, "eastward_wind"))
  call handle_err(nf_put_att_text(ncid, u_id, "units", len(metadata%wind_units), metadata%wind_units))
  call handle_err(nf_put_att_real(ncid, u_id, "_FillValue", NF_REAL, 1, 99.9))
  call handle_err(nf_put_att_real(ncid, u_id, "missing_value", NF_REAL, 1, 99.9))

  call handle_err(nf_put_att_text(ncid, uerr_id, "standard_name", 19, "eastward_wind_error"))
  call handle_err(nf_put_att_text(ncid, uerr_id, "units", len(metadata%wind_units), metadata%wind_units))
  call handle_err(nf_put_att_real(ncid, uerr_id, "_FillValue", NF_REAL, 1, 99.9))
  call handle_err(nf_put_att_real(ncid, uerr_id, "missing_value", NF_REAL, 1, 99.9))

  call handle_err(nf_put_att_text(ncid, v_id, "standard_name", 14, "northward_wind"))
  call handle_err(nf_put_att_text(ncid, v_id, "units", len(metadata%wind_units), metadata%wind_units))
  call handle_err(nf_put_att_real(ncid, v_id, "_FillValue", NF_REAL, 1, 99.9))
  call handle_err(nf_put_att_real(ncid, v_id, "missing_value", NF_REAL, 1, 99.9))

  call handle_err(nf_put_att_text(ncid, verr_id, "standard_name", 20, "northward_wind_error"))
  call handle_err(nf_put_att_text(ncid, verr_id, "units", len(metadata%wind_units), metadata%wind_units))
  call handle_err(nf_put_att_real(ncid, verr_id, "_FillValue", NF_REAL, 1, 99.9))
  call handle_err(nf_put_att_real(ncid, verr_id, "missing_value", NF_REAL, 1, 99.9))

  call handle_err(nf_put_att_text(ncid, NF_GLOBAL, "platform", len(trim(adjustl(metadata%platform))), &
       trim(adjustl(metadata%platform))))

  write ( position_str, '(F9.4,A1,F8.4)' ) position_start_lon, ' ', position_start_lat 
  call handle_err(nf_put_att_text(ncid, NF_GLOBAL, "launch_start_position", 18, position_str))

  write ( position_str, '(F9.4,A1,F8.4)' ) position_end_lon, ' ', position_end_lat 
  call handle_err(nf_put_att_text(ncid, NF_GLOBAL, "launch_end_position", 18, position_str))

  call handle_err(nf_enddef (ncid))

  time(1) = 0.0

  call handle_err(nf_put_vara(ncid, launch_time_id, start(2), edge(2), time))

  call handle_err(nf_put_vara(ncid, flight_time_id, start, edge, radiosondedata(1:no_of_levels)%time_aft_launch))
  call handle_err(nf_put_vara(ncid, p_id,           start, edge, radiosondedata(1:no_of_levels)%pressure))
  call handle_err(nf_put_vara(ncid, alt_id,         start, edge, radiosondedata(1:no_of_levels)%altitude))
  call handle_err(nf_put_vara(ncid, t_id,           start, edge, radiosondedata(1:no_of_levels)%temperature))
  call handle_err(nf_put_vara(ncid, terr_id,        start, edge, radiosondedata(1:no_of_levels)%temp_error))
  call handle_err(nf_put_vara(ncid, sh_id,          start, edge, radiosondedata(1:no_of_levels)%specific_humdty))
  call handle_err(nf_put_vara(ncid, sherr_id,       start, edge, radiosondedata(1:no_of_levels)%humdty_error))
  call handle_err(nf_put_vara(ncid, u_id,           start, edge, radiosondedata(1:no_of_levels)%wind_vel_u_comp))
  call handle_err(nf_put_vara(ncid, uerr_id,        start, edge, radiosondedata(1:no_of_levels)%u_comp_error))
  call handle_err(nf_put_vara(ncid, v_id,           start, edge, radiosondedata(1:no_of_levels)%wind_vel_v_comp))
  call handle_err(nf_put_vara(ncid, verr_id,        start, edge, radiosondedata(1:no_of_levels)%v_comp_error))

  call handle_err(nf_close(ncid))

end subroutine write_netcdf

! ----------------------

subroutine handle_err(status)
  implicit none
  include 'netcdf.inc'

  integer, intent(in) :: status

  if (status .ne. nf_noerr) then
     print *, nf_strerror(status)
     stop
  endif
end subroutine handle_err
