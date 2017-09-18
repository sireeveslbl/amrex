
! ::: -----------------------------------------------------------
! ::: This routine will tag high error cells based on the state
! ::: 
! ::: INPUTS/OUTPUTS:
! ::: 
! ::: tag        <=  integer tag array
! ::: tag_lo,hi   => index extent of tag array
! ::: state       => state array
! ::: state_lo,hi => index extent of state array
! ::: set         => integer value to tag cell for refinement
! ::: clear       => integer value to untag cell
! ::: lo,hi       => work region we are allowed to change
! ::: dx          => cell size
! ::: problo      => phys loc of lower left corner of prob domain
! ::: time        => problem evolution time
! ::: level       => refinement level of this array
! ::: -----------------------------------------------------------

subroutine state_error(tag,tag_lo,tag_hi, &
                       state,state_lo,state_hi, &
                       set,clear,&
                       lo,hi,&
                       dx,problo,time,phierr) bind(C, name="state_error")

  implicit none
  
  integer          :: lo(3),hi(3)
  integer          :: state_lo(3),state_hi(3)
  integer          :: tag_lo(3),tag_hi(3)
  double precision :: state(state_lo(1):state_hi(1), &
                            state_lo(2):state_hi(2), &
                            state_lo(3):state_hi(3))
  integer          :: tag(tag_lo(1):tag_hi(1),tag_lo(2):tag_hi(2),tag_lo(3):tag_hi(3))
  double precision :: problo(3),dx(3),time,phierr
  integer          :: set,clear

  integer          :: i, j, k
  double precision :: center(3)

  ! Tag on regions of high phi
  do       k = lo(3), hi(3)
     center(3) = problo(3) + (k-0.5)*dx(3)
     do    j = lo(2), hi(2)
        center(2) = problo(2) + (j-0.5)*dx(2)
        do i = lo(1), hi(1)
           center(1) = problo(1) + (i-0.5)*dx(1)
           if ( &
                (abs(center(3) - 0.0625) < dx(3) .or. & 
                 abs(center(3) - 0.1875) < dx(3) .or. &
                 abs(center(3) - 0.3125) < dx(3) .or. &
                 abs(center(3) - 0.4375) < dx(3) .or. &
                 abs(center(3) - 0.5625) < dx(3) .or. &
                 abs(center(3) - 0.6875) < dx(3) .or. &
                 abs(center(3) - 0.8125) < dx(3) .or. &
                 abs(center(3) - 0.9375) < dx(3) ) &
                 .and. &
                (abs(center(2) - 0.0625) < dx(2) .or. & 
                 abs(center(2) - 0.1875) < dx(2) .or. &
                 abs(center(2) - 0.3125) < dx(2) .or. &
                 abs(center(2) - 0.4375) < dx(2) .or. &
                 abs(center(2) - 0.5625) < dx(2) .or. &
                 abs(center(2) - 0.6875) < dx(2) .or. &
                 abs(center(2) - 0.8125) < dx(2) .or. &
                 abs(center(2) - 0.9375) < dx(2) ) &
                 .and. &
                (abs(center(1) - 0.0625) < dx(1) .or. & 
                 abs(center(1) - 0.1875) < dx(1) .or. &
                 abs(center(1) - 0.3125) < dx(1) .or. &
                 abs(center(1) - 0.4375) < dx(1) .or. &
                 abs(center(1) - 0.5625) < dx(1) .or. &
                 abs(center(1) - 0.6875) < dx(1) .or. &
                 abs(center(1) - 0.8125) < dx(1) .or. &
                 abs(center(1) - 0.9375) < dx(1) ) &
              ) then
                tag(i,j,k) = set
           endif
        enddo
     enddo
  enddo

end subroutine state_error

