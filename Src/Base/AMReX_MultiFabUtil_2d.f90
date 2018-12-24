
! ***************************************************************************************
! subroutine bl_avgdown_faces
! ***************************************************************************************

subroutine bl_avgdown_faces (lo, hi, &
     f, f_l1, f_l2, f_h1, f_h2, &
     c, c_l1, c_l2, c_h1, c_h2, &
     ratio, idir,nc)

  use amrex_fort_module, only : amrex_real
  implicit none
  integer          :: lo(2),hi(2)
  integer          :: f_l1, f_l2, f_h1, f_h2
  integer          :: c_l1, c_l2, c_h1, c_h2
  integer          :: ratio(3), idir,nc
  real(amrex_real) :: f(f_l1:f_h1, f_l2:f_h2, nc)
  real(amrex_real) :: c(c_l1:c_h1, c_l2:c_h2, nc)

  ! Local variables
  integer i,j,n,facx,facy,iref,jref, ii, jj
  real(amrex_real) :: facInv

  facx = ratio(1)
  facy = ratio(2)

  if (idir .eq. 0) then

     facInv = 1.d0 / facy

     do n = 1, nc
        do j     = lo(2), hi(2)
           jj    = j * facy
           do i  = lo(1), hi(1)
              ii = i * facx
              c(i,j,n) = 0.d0
              do jref = 0, facy-1
                 c(i,j,n) = c(i,j,n) + f(ii,jj+jref,n)
              end do
              c(i,j,n) = c(i,j,n) * facInv
           end do
        end do
     end do

  else 

     facInv = 1.d0 / facx

     do n = 1, nc
        do j     = lo(2), hi(2)
           jj    = j * facy
           do i  = lo(1), hi(1)
              ii = i * facx
              c(i,j,n) = 0.d0
              do iref = 0,facx-1
                 c(i,j,n) = c(i,j,n) + f(ii+iref,jj,n)
              end do
              c(i,j,n) = c(i,j,n) *facInv
           end do
        end do
     end do

  end if

end subroutine bl_avgdown_faces

! ***************************************************************************************
! subroutine bl_avgdown_edges
! ***************************************************************************************

subroutine bl_avgdown_edges (lo, hi, &
     f, f_l1, f_l2, f_h1, f_h2, &
     c, c_l1, c_l2, c_h1, c_h2, &
     ratio, idir,nc)

  use amrex_fort_module, only : amrex_real
  implicit none
  integer          :: lo(2),hi(2)
  integer          :: f_l1, f_l2, f_h1, f_h2
  integer          :: c_l1, c_l2, c_h1, c_h2
  integer          :: ratio(3), idir,nc
  real(amrex_real) :: f(f_l1:f_h1, f_l2:f_h2, nc)
  real(amrex_real) :: c(c_l1:c_h1, c_l2:c_h2, nc)

  ! Local variables
  integer i,j,n,facx,facy,iref,jref, ii, jj
  real(amrex_real) :: facInv

  facx = ratio(1)
  facy = ratio(2)

  if (idir .eq. 0) then

     facInv = 1.d0 / facx

     do n = 1, nc
        do j     = lo(2), hi(2)
           jj    = j * facy
           do i  = lo(1), hi(1)
              ii = i * facx
              c(i,j,n) = 0.d0
              do iref = 0, facx-1
                 c(i,j,n) = c(i,j,n) + f(ii+iref,jj,n)
              end do
              c(i,j,n) = c(i,j,n) * facInv
           end do
        end do
     end do

  else 

     facInv = 1.d0 / facy

     do n = 1, nc
        do j     = lo(2), hi(2)
           jj    = j * facy
           do i  = lo(1), hi(1)
              ii = i * facx
              c(i,j,n) = 0.d0
              do jref = 0,facy-1
                 c(i,j,n) = c(i,j,n) + f(ii,jj+jref,n)
              end do
              c(i,j,n) = c(i,j,n) *facInv
           end do
        end do
     end do

  end if

end subroutine bl_avgdown_edges

! ***************************************************************************************
! subroutine bl_avgdown - THIS VERISON DOES NOT DO VOLUME WEIGHTING
! ***************************************************************************************

subroutine bl_avgdown (lo,hi,&
     fine,f_l1,f_l2,f_h1,f_h2, &
     crse,c_l1,c_l2,c_h1,c_h2, &
     lrat,ncomp)
  
  use amrex_fort_module, only : amrex_real
  implicit none
  
  integer f_l1,f_l2,f_h1,f_h2
  integer c_l1,c_l2,c_h1,c_h2
  integer lo(2), hi(2)
  integer lrat(2), ncomp
  real(amrex_real) fine(f_l1:f_h1,f_l2:f_h2,ncomp)
  real(amrex_real) crse(c_l1:c_h1,c_l2:c_h2,ncomp)

  integer :: i, j, ii, jj, n, iref, jref
  real(amrex_real) :: volfrac

  volfrac = 1.d0 / dble(lrat(1)*lrat(2))

  do n = 1, ncomp
     do j     = lo(2), hi(2)
        jj    = j * lrat(2)
        do i  = lo(1), hi(1)
           ii = i * lrat(1)
           crse(i,j,n) = 0.d0
           do    jref = 0, lrat(2)-1
              do iref = 0, lrat(1)-1
                 crse(i,j,n) = crse(i,j,n) + fine(ii+iref,jj+jref,n)
              end do
           end do
           crse(i,j,n) = volfrac * crse(i,j,n)
        end do
     end do
  end do

end subroutine bl_avgdown

! ***************************************************************************************
! subroutine bl_avgdown_nodes
! ***************************************************************************************

subroutine bl_avgdown_nodes (lo,hi,&
     fine,f_l1,f_l2,f_h1,f_h2, &
     crse,c_l1,c_l2,c_h1,c_h2, &
     lrat,ncomp)
  
  use amrex_fort_module, only : amrex_real
  implicit none
  
  integer f_l1,f_l2,f_h1,f_h2
  integer c_l1,c_l2,c_h1,c_h2
  integer lo(2), hi(2)
  integer lrat(2), ncomp
  real(amrex_real) fine(f_l1:f_h1,f_l2:f_h2,ncomp)
  real(amrex_real) crse(c_l1:c_h1,c_l2:c_h2,ncomp)

  integer :: i, j, ii, jj, n

  do n = 1, ncomp
     do j     = lo(2), hi(2)
        jj    = j * lrat(2)
        do i  = lo(1), hi(1)
           ii = i * lrat(1)
           crse(i,j,n) = fine(ii, jj, n)
        end do
     end do
  end do

end subroutine bl_avgdown_nodes

! ***************************************************************************************
! subroutine bl_avgdown_with_vol
! ***************************************************************************************

subroutine bl_avgdown_with_vol (lo,hi,&
     fine,f_l1,f_l2,f_h1,f_h2, &
     crse,c_l1,c_l2,c_h1,c_h2, &
     fv,fv_l1,fv_l2,fv_h1,fv_h2, &
     lrat,ncomp)

  use amrex_fort_module, only : amrex_real
  implicit none
  
  integer f_l1,f_l2,f_h1,f_h2
  integer c_l1,c_l2,c_h1,c_h2
  integer fv_l1,fv_l2,fv_h1,fv_h2
  integer lo(2), hi(2)
  integer lrat(2), ncomp
  real(amrex_real) fine(f_l1:f_h1,f_l2:f_h2,ncomp)
  real(amrex_real) crse(c_l1:c_h1,c_l2:c_h2,ncomp)
  real(amrex_real) fv(fv_l1:fv_h1,fv_l2:fv_h2)

  integer :: i, j, ii, jj, n, iref, jref
  real(amrex_real) :: cv

  do n = 1, ncomp
     do j     = lo(2), hi(2)
        jj    = j * lrat(2)
        do i  = lo(1), hi(1)
           ii = i * lrat(1)
           crse(i,j,n) = 0.d0
           cv          = 0.d0
           do    jref = 0, lrat(2)-1
              do iref = 0, lrat(1)-1
                 cv          = cv          +                         fv(ii+iref,jj+jref)
                 crse(i,j,n) = crse(i,j,n) + fine(ii+iref,jj+jref,n)*fv(ii+iref,jj+jref)
              end do
           end do
           crse(i,j,n) = crse(i,j,n) / cv
        end do
     end do
  end do

end subroutine bl_avgdown_with_vol


subroutine amrex_compute_divergence (lo, hi, divu, dlo, dhi, u, ulo, uhi, &
     v, vlo, vhi, dxinv) bind(c)
  use amrex_fort_module, only : amrex_real
  implicit none
  integer, dimension(2), intent(in) :: lo, hi, dlo, dhi, ulo, uhi, vlo, vhi
  real(amrex_real), intent(inout) :: divu(dlo(1):dhi(1),dlo(2):dhi(2))
  real(amrex_real), intent(in   ) ::    u(ulo(1):uhi(1),ulo(2):uhi(2))
  real(amrex_real), intent(in   ) ::    v(vlo(1):vhi(1),vlo(2):vhi(2))
  real(amrex_real), intent(in) :: dxinv(2)
  integer :: i,j
  do    j = lo(2), hi(2)
     do i = lo(1), hi(1)
        divu(i,j) = dxinv(1) * (u(i+1,j)-u(i,j)) + dxinv(2) * (v(i,j+1)-v(i,j))
     end do
  end do
end subroutine amrex_compute_divergence
