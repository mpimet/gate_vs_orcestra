#define FILENAME_LENGHT 132
module GATEdship_mod

  real, parameter :: lon_min = -106.0
  real, parameter :: lon_max = 62.0
  real, parameter :: lat_min = -22.0
  real, parameter :: lat_max = 38.0
  real, parameter :: res = 0.5

  ! Calculate the number of grid points in each direction
  integer, parameter :: nlon = int((lon_max - lon_min) / res) + 1
  integer, parameter :: nlat = int((lat_max - lat_min) / res) + 1

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
     integer :: SST
  end type GATE_dship_type

  type :: GATE_metadata_type

     character(len=32) :: shipname
     type (datetime)   :: measurement_time
     real              :: interval
     character         :: interval_unit

     character(len=4)  :: temperature_unit='degC'

  end type GATE_metadata_type

  public :: GATE_dship_type, GATE_metadate_type

end module GATEdship_mod

! ----------------------

program GATEdship

  ! ASCII Input
  !
  !     3.00.02.104-3.31.02.101_19740601-19740930/
  !  GATE_AND_COMM_SHIPS
  !

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
 
     if ( line(2:33) /= "(I1,I4,I10,I5,33(10I3),7I3,889X)" ) then
        write ( * , * )
        write ( * , * ) trim(infile), ':'
        write ( * , * ) " - wrong format description"
     end if

     write ( * , * )
     write ( * , * ) 'now processing ', trim(infile)
     call convert_data ( infile )

  case default
     close (unit=10)
     write ( * , * )
     write ( * , * ) ' ', trim(infile), ' does not contain data'

  end select

end program GATEdship

! ----------------------

subroutine convert_data (infile)

  use, intrinsic :: iso_fortran_env, only : iostat_end
  use GATEdship_mod

  implicit none

  character(len=FILENAME_LENGHT), intent(in) :: infile

  ! 4 ints plus 337 measurements per full data line (see format string)

  integer :: type_id          ! file header record type indicator
  integer :: records_in_line  ! number of of records contained in one full data line
  integer :: records_handled  ! number of records allready stored away
  integer :: line_number      ! of one line containing full format string

  type (GATE_dship_type) :: shipdata(nlon)

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

  dshipsize_inc = 11*nlon
  dshipsize     = dshipsize_inc

  allocate (dshipcurrent(dshipsize))
  dshipdata => dshipcurrent

  open (unit=10, file=trim(infile), status='old', form='formatted', action='read')

  ! file section 1, summary of measurement

  do i = 1, 24

     read( 10, 100, iostat=ierror ) line(1:80)

     select case (ierror)

     case ( 0 )

        metadata%interval = 1
        metadata%interval_unit = "D"
     
        if ( i == 4 ) then
           read(line(2:15),'(i4,5i2)')                  &
              metadata%measurement_time%year,   &
              metadata%measurement_time%month,  &
              metadata%measurement_time%day,    &
              metadata%measurement_time%hour,   &
              metadata%measurement_time%minute, &
              metadata%measurement_time%second

           metadata%measurement_time%hour = 12
           metadata%measurement_time%minute = 0
           metadata%measurement_time%second = 0

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

        read(line, 110, iostat=ierror ) type_id, records_in_line, records_handled, line_number, shipdata

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
        
        do i = 1, nlon
           no_of_measurement = no_of_measurement + 1
           dshipdata(no_of_measurement) = shipdata(i)
        end do

     end if

  end do ! while-loop

  close (10)

  write ( * , * ) 'Processed ', no_of_measurement, 'data records.'

  call write_netcdf ( infile, no_of_measurement, dshipdata, metadata )

  deallocate ( dshipcurrent )

110 format(I1,I4,I10,I5,337I3,889X)
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

  character(len = 8)  :: clockdate
  character(len = 10) :: clocktime
  character(len = 5)  :: timezone
  integer             :: values(8)

  character(len = 256) :: history

  ! Define the time dimension
  integer, parameter :: ntime = 1
  integer            :: start(1)
  integer            :: count(1)
  real, dimension(ntime) :: time

  ! Define the grid coordinates
  real, dimension(nlon) :: lon
  real, dimension(nlat) :: lat

  real, dimension(nlon, nlat, ntime) :: sst
  real, parameter :: fill_value = -9999.0

  ! Define the NetCDF ids
  integer :: ncid, lon_id, lat_id, time_id
  integer :: lat_dimid, lon_dimid, time_dimid
  integer :: sst_id

  integer :: dimids(3)

  integer :: status

  character(len=33) :: seconds_since

  integer :: i, j

  ! some preparation for writing global attributes

  call date_and_time(clockdate, clocktime, timezone, values)

  write (history, "(A,I4,A1,I0.2,A1,I0.2,A1,3(I0.2,A1),A)" ) &
                      "Created by Rene Redler, MPI-M on ",   &
                      values(1), "-", values(2), "-", values(3), " ", &
                      values(5), ":", values(6), ":", values(7), " ", &
                      "from GATE_AND_COMM_SHIPS mapped sst data in archive directory 3.00.02.104-3.31.02.101_19740601-19740930"

  write ( seconds_since , '(A14,I4,A1,2(I2.2,A1),2(I2.2,A1),I2.2)' ) &
       & 'seconds since ',                        &
       & metadata%measurement_time%year,   '-',  &
       & metadata%measurement_time%month,  '-',  &
       & metadata%measurement_time%day,    ' ',  &
       & metadata%measurement_time%hour,   ':',  &
       & metadata%measurement_time%minute, ':',  &
       & metadata%measurement_time%second

  ! start writing

  write ( * , * ) trim(infile), ':'
  write ( outfile, '(A,A3)' ) trim(infile), '.nc'

  call handle_err(nf_create( outfile, NF_CLOBBER, ncid))

  ! Define the axis

  if ( nlon /= 337 ) then
    write ( *, * ) "Error in nlon"
    stop 
  endif

  if ( nlat /= 121 ) then
    write ( *, * ) "Error in nlat"
    stop 
  endif

  call handle_err(nf_def_dim(ncid, 'longitude', nlon, lon_dimid))
  call handle_err(nf_def_var(ncid, 'longitude', NF_FLOAT, 1, (/lon_dimid/), lon_id))
  call handle_err(nf_put_att(ncid, lon_id, 'units', NF_CHAR, 12, 'degrees_east'))
  call handle_err(nf_put_att(ncid, lon_id, 'long_name', NF_CHAR, 9, 'longitude'))
  call handle_err(nf_put_att(ncid, lon_id, 'standard_name', NF_CHAR, 9, 'longitude'))

  call handle_err(nf_def_dim(ncid, 'latitude', nlat, lat_dimid))
  call handle_err(nf_def_var(ncid, 'latitude', NF_FLOAT, 1, (/lat_dimid/), lat_id))
  call handle_err(nf_put_att(ncid, lat_id, 'units', NF_CHAR, 13, 'degrees_north'))
  call handle_err(nf_put_att(ncid, lat_id, 'long_name', NF_CHAR, 8, 'latitude'))
  call handle_err(nf_put_att(ncid, lat_id, 'standard_name', NF_CHAR, 8, 'latitude'))

  call handle_err(nf_def_dim(ncid, 'time', 1, time_dimid))
  call handle_err(nf_def_var(ncid, "time", NF_REAL, 1, (/ time_dimid /), time_id))
  call handle_err(nf_put_att(ncid, time_id, "units", NF_CHAR, len(seconds_since), seconds_since))

  dimids(1) = time_dimid
  dimids(2) = lon_dimid
  dimids(3) = lat_dimid

  call handle_err(nf_def_var(ncid, "sst", NF_FLOAT, 3, dimids, sst_id))

  call handle_err(nf_put_att_text(ncid, sst_id, "standard_name", 24, "sea_surface_temperature"))
  call handle_err(nf_put_att_text(ncid, sst_id, "units", len(metadata%temperature_unit), metadata%temperature_unit))
  call handle_err(nf_put_att_real(ncid, sst_id, "_FillValue", NF_REAL, 1, fill_value))

  call handle_err(nf_put_att_text(ncid, time_dimid, 'units', len(seconds_since), seconds_since))
  call handle_err(nf_put_att_text(ncid, time_dimid, "calendar", 19, "proleptic_gregorian"))

  call handle_err(nf_put_att_text(ncid, NF_GLOBAL, "history", len(trim(history)), history))

  ! End the definition mode
  call handle_err(nf_enddef(ncid))

  ! Initialize the grid coordinates
  do i = 1, nlon
    lon(i) = lon_min + (i - 1) * res
  end do

  do j = 1, nlat
    lat(j) = lat_min + (j - 1) * res
  end do

  do j = 1, nlat
     do i = 1, nlon
        if ( dshipdata((j-1)*nlon+i)%sst == 0 ) then
           sst(i,j,1) = fill_value
        else
           sst(i,j,1) = real(dshipdata((j-1)*nlon+i)%sst)/10.0
        endif
     enddo
  enddo

  time = 0.0

  start(1) = 1
  count(1) = 1
  call handle_err(nf_put_vara_real(ncid, time_id, start, count, time))

  ! Write the grid coordinates to the NetCDF file
  call handle_err(nf_put_var_real(ncid, lon_id, lon))
  call handle_err(nf_put_var_real(ncid, lat_id, lat))

  ! Write the data
  call handle_err(nf_put_var_real(ncid, sst_id, sst))

  ! Close the file
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
