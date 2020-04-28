pro wip_data2dem_reg, logT ,TRmatrix ,data ,edata ,$
  mint=mint,maxt=maxt,nt=nt,$
  order=order , guess=guess ,reg_tweak=reg_tweak,$
  channels=channels,debug=debug

  ; 28-Apr-2020 IGH   wip version that debugs to check what is happening

  order=0
  guess=0
  reg_tweak=1.
  nt=33
  mint=5.7
  maxt=7.3
  pos=0

  ; Just put the data directly in here to check what is happening internally
  restore,file='aia_resp.dat'
  filt=[0,1,2,3,4,6]
  data=[274.2, 166.9, 2842.9, 8496.3, 6941.9, 876.1]
  nf=n_elements(data)
  edata=[14.9, 10.9, 41.1, 65.3, 47.1, 14.1]
  TRmatrix=tresp.all[*,filt]
  logt=tresp.logte
  channels=tresp.channels[filt]


  if (nf ge nt) then print, '!!!!! Warning: n_T must be bigger than n_F'
  if (nf ne n_elements(edata)) then print, '!!!!! Warning: Data and eData must be the same size'
  if ((where(edata le 0.))[0] ne -1) then print, '!!!!! Warning: None of eData can be 0'

  ;interpolate temperature responses to binning desired of DEM
  logTint=mint+(maxt-mint)*findgen(nt)/(nt-1.0)
  TRmatint=dblarr(nt,nf)
  ;better interpolation if done in log space though need check not -NaN
  for i=0, nf-1 do TRmatint[*,i]=10d^interpol(alog10(TRmatrix[*,i]), logT, logTint) > 0.
  ; also get response in correct units and scale to make numerics simplier
  dlogTint  =logTint[1:nT-1]-logTint[0:nT-2]
  dlogTint=[dlogTint,dlogTint[nt-2]]

  ;Before we begin can work out the conversion factors for DEM to EM
  lgt_edg=fltarr(nt-1)
  for ll=0,nt-2 do lgt_edg[ll]=(logTint[ll+1]+logTint[ll])*0.5
  lgt_edg=[lgt_edg[0]-(lgt_edg[1]-lgt_edg[0]),lgt_edg,lgt_edg[nt-2]+(lgt_edg[nt-2]-lgt_edg[nt-3])]
  dlgT_edg=lgt_edg[1:nt]-lgt_edg[0:nt-1]
  DEMtoEM=10d^lgt_edg*alog(10d^dlgt_edg)

  ; Now intrepolate the response functions to the temperature binning of DEM output
  RMatrix=dblarr(nT,nF)
  for i=0, nF-1  do RMatrix[*,i]=TRmatint[*,i]*10d^logTint*alog(10d^dlogTint)
  RMatrix=RMatrix*1d20
  RMatrix_org=Rmatrix
  DEM_model =fltarr(nt)

  ; normalize everything by the errors
  data_in=data/edata
  edata_in=edata/edata
  for i=0, nf-1 do RMatrix[*,i]=RMatrix[*,i]/edata[i]

  ;********************************************************************************************************
  ; Regularization is run twice
  ; the first time the constraint matrix is taken as the identity matrix normalized by dlogT
  ; must use this 0th order as no dem_model guess
;  L=fltarr(nT,nT)
  L=dblarr(nT,nT)
  for i=0, nT-1 do L[i,i]=1.0/sqrt(dlogTint[i])

  ; GSVD on temperature responses (Rmatrix) and constraint matrix (L)
  dem_inv_gsvdcsq,RMatrix,L,Alpha,Betta,U,V,W

  print,'RMatrix ',size(rmatrix,/dim)
  print,'L ',size(L,/dim)
  print,'Alpha ',size(Alpha,/dim)
  print,'Betta ',size(Betta,/dim)
  print,'U ',size(U,/dim)
  print,'V ',size(V,/dim)
  print,'W ',size(W,/dim)

  diaga=diag_matrix(alpha)
  diagb=diag_matrix(betta)

  print,'alpha^2+beta^2 = 1 ? ',alpha^2+betta^2

  r2=U ## diaga ## invert(W,/double)
  l2=V ## diagb ## invert(W,/double)

  ; L is only a float here ?????
  print,'L: '
  print,l(indgen(nt),indgen(nt))
  print,'V ## diagb ## W^-1: '
  print,l2(indgen(nt),indgen(nt))
  print,'R, U ## diaga ## W^-1: '
  for i=0,nf-1 do print,rmatrix[*,i],r2[*,i]
  
  

  stop

  ; Determine the regularization parameter
  ; for first run using weekly regularized with reg_tweak=sqt(nt)
  dem_inv_reg_parameter,Alpha,Betta,U,W,data_in,edata_in,transpose(DEM_model)*Guess,sqrt(nt*1.0),opt

  ; Now work out the regularized DEM solution
  dem_inv_reg_solution,Alpha,Betta,U,W,data_in,opt,DEM_model*Guess,DEM_reg
  ;********************************************************************************************************
  ; For second run use found regularized solution as weighting for constraint matrix and possible guess solution
  DEM_reg=DEM_reg *(DEM_reg GT 0)+1e-4*max(Dem_reg)*(DEM_reg LT 0)
  DEM_reg=smooth(DEM_reg,3)
  DEM_model=DEM_reg


  ;This time make constraint to specified order
  dem_inv_make_constraint,L,logTint,dlogTint,DEM_model,order

  ; GSVD on temperature responses (Rmatrix) and constraint matrix (L)
  dem_inv_gsvdcsq,RMatrix,L,Alpha,Betta,U,V,W


  ; Here do not require positive solution so first find regularization parameter than gives chosen chi^2
  ; Then use it to work out the regularized solution

  ; Determine the regularization parameter
  ; for second run regularize to level specified with reg_tweak
  dem_inv_reg_parameter,Alpha,Betta,U,W,data_in,edata_in,transpose(DEM_model)*Guess,reg_tweak,opt

  ; Now work out the regularized DEM solution
  dem_inv_reg_solution,Alpha,Betta,U,W,data_in,opt,DEM_model*Guess,DEM_reg

  ;********************************************************************************************************
  ; now work out the temperature resolution/horizontal spread
  dem_inv_reg_resolution,Alpha,Betta,opt,W,logTint,dlogTint,FWHM,cent,RK,fwhm2

  ; now work out the DEM error (vertical error)
  npass=300.
  dem_inv_confidence_interval,DEM_reg,data_in,edata_in,Alpha,Betta,U,W,opt,DEM_model,Guess,Npass,reg_sol_err
  ;********************************************************************************************************
  ;Calculate the data signal that found regularized DEM gives you and work out data residuals and chisq
  data_reg=transpose(Rmatrix_org##dem_reg)
  residuals=(data-data_reg)/edata
  chisq=total(residuals^2)/(nf*1.0)

  data_cont_t=fltarr(nt,nf)
  for i=0, nf-1 do data_cont_t[*,i]=Rmatrix_org[*,i]*dem_reg

  reg_solution={data:data,edata:edata, Tresp:Trmatint,channels:channels,$
    DEM:DEM_reg[0:nT-1]*1d20, eDEM:reg_sol_err[0:nT-1]*1d20, $
    logT:logTint,elogT:fwhm/(2.0*sqrt(2*alog(2))),RK:RK,$
    data_reg:data_reg,residuals:residuals,chisq:chisq,data_cont_t:data_cont_t,$
    reg_tweak:reg_tweak,guess:guess,order:order,DEMtoEM:DEMtoEM}


  ;  return,reg_solution

  if keyword_set(debug) then stop

end

