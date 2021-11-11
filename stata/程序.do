



use 数据.dta, clear

set matsize 11000

ta year,gen(yrid)
xtset stkcd year
global v0 "size age leverage SOE cash roa topten mancomp board_scale sale exprt rdsy nrdsy proft rd_scale tangible tobin indept ins  tfp_ols  compete compete2 gdpr ipp hcap finance mkt gdppc third"
global fixeff "yrid2-yrid8"

*********表1 *************
  
tabstat patenta_all $v0, stat(N mean sd ) c(s) 

*********表 2 ***********************************
* 其他表格类似
est clear
global yvar "patenta_all" 
global pva=1 
global pcv=0.05 
global v2 "size age leverage SOE cash roa topten mancomp board_scale sale exprt rdsy nrdsy proft rd_scale tangible tobin indept ins  tfp_ols  compete compete2 gdpr ipp hcap finance mkt gdppc third " 
global v1 "$v2 $fixeff" 

xtscc $yvar $v1  , fe  lag(2)

***********************************************
quietly {
mat yyy=e(b)
global NM=e(N) 
global NP=colsof(yyy) 
tempvar esamp
gen `esamp'=e(sample)
eststo

mat stdtp=J(1,$NP,0) 

mat sxsx=r(table)
forvalues i=1/$NP{
  mat stdtp[1,`i']=sxsx[4,`i'] 
}

cap drop res_m
predict res_m,res 
qui su res_m
local var_res=r(sd)*r(sd)

mat rq=yyy
local i=1
local mname ""
foreach var of varlist $v1 {
  tempvar t_`var'
  gen `t_`var''=_b[`var']*`var'
  qui su `t_`var'' if `esamp'==1
  mat rq[1,`i']=r(sd)*r(sd)
  if $pva ==1&stdtp[1,`i']>$pcv{ 
    mat rq[1,`i']= 0
  }
  local i=`i'+1
  local mname "`mname' `var'"
}
mat rq[1,$NP] =`var_res'
local var_sum=0
forvalues i=1/$NP{
  local var_sum=`var_sum'+rq[1,`i']
}
mat rq= rq/`var_sum'*100
matrix colnames rq=`mname' Residual 
mat m_var= rq
eststo:ereturn post rq


mat rwq=yyy
local i=1
foreach var of varlist $v1 {
  qui su `var' if `esamp'==1
  mat rwq[1,`i']= abs(_b[`var']*r(mean))
  if $pva ==1&stdtp[1,`i']>$pcv{ 
    mat rwq[1,`i']= 0
  }
  local i=`i'+1
}
mat rwq[1,$NP] =abs(yyy[1,$NP])
local var_sum=0
forvalues i=1/$NP{
  local var_sum=`var_sum'+rwq[1,`i']
}
mat rwq= rwq/`var_sum'*100
mat m_level=rwq
eststo:ereturn post rwq


mat rqm=J(1,$NP+1,0)
local i=1
local mname ""
foreach var of varlist $v1 {
  qui su `var'
  mat rqm[1,`i']= (m_var[1,`i']+m_level[1,`i'])/2
  local i=`i'+1
  local mname "`mname' `var'"
}
mat rqm[1,$NP] = m_level[1,$NP]/2 
mat rqm[1,$NP+1] = m_var[1,$NP]/2  
matrix colnames rqm=`mname' _cons Residual
eststo:ereturn post rqm
}

esttab using Table.rtf ,  scalar(r2) star( ** 0.05 *** 0.01) compress b(3) se(2) mtitles(reg variance level average) nogap   onecell replace 




 

 

 