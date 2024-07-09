library(circular)
library(knitr)

#### aproachces above a time threshold
df<-read.csv('tables/first_approach_raw_day_2_thr_0.4.csv')


## Select a subset of the distribution
thr_1_time=0
thr_2_time=100000
n_samples=1000
angle_filter=pi

# Fro raw and mat
tg.approach <- subset(df, behaviour == "approach" & phenotype=="tg"  & ((diff_time>thr_1_time)  & (diff_time<thr_2_time))
                      & (angle_rad_from_complex>-angle_filter) & (angle_rad_from_complex<angle_filter))
wt.approach <- subset(df, behaviour == "approach" & phenotype=="wt"  & ((diff_time>thr_1_time)  & (diff_time<thr_2_time))
                      & (angle_rad_from_complex>-angle_filter) & (angle_rad_from_complex<angle_filter))


################       For raw MATRIX ################################
###########################################################################
# #
tg.angle=circular(tg.approach$angle_rad_from_complex ,units="radians",template = "none")
wt.angle=circular(wt.approach$angle_rad_from_complex ,units="radians",template = "none")

# # #####                       EXPLORATION
# tg.angle=circular(tg.exploration$angle_rad_from_complex ,units="radians",template = "none")
# wt.angle=circular(wt.exploration$angle_rad_from_complex ,units="radians",template = "none")
## stastistics
wt.mean=mean.circular(wt.angle);wt.mean
tg.mean=mean.circular(tg.angle);tg.mean
wt.median=median.circular(wt.angle);wt.median
tg.median=median.circular(tg.angle);tg.median
## meassures of disperson
# The magnitude of the sum off all complex vectors
wt.magnitude=rho.circular(wt.angle);wt.magnitude
tg.magnitude=rho.circular(tg.angle);tg.magnitude
wt.circular_variance=1-wt.magnitude;wt.circular_variance
tg.circular_variance=1-tg.magnitude;tg.circular_variance
to_plot<-tg.angle
kernel_band<-40
shrink<-1.6
cex=0.9
prop=1.8
sep=0.06
bins=21
plot.circular(to_plot,stack = TRUE,pch = 20, sep = sep, start.sep=0.04,shrink = shrink
              ,main='tg',zero=pi/2,cex=cex,axes=FALSE)
# Here we have adopted the default convention
# of the rose.diag function where the radius of a segment is taken to be the square root of
# the relative frequency.
# The
# other convention is to use a radius which is linearly related to the relative frequency
# his second convention can be implemented
# by using the modifier radii.scale = linear.
rose.diag(to_plot, bins=bins, col="deepskyblue2", cex=1.5, prop=prop,
          add=TRUE,axes=FALSE,zero=pi/2)
# lines(density.circular(to_plot, bw=kernel_band),zero=pi/2, lwd=2, lty=3)
axis.circular(at=circular(seq(0, 7*pi/4,pi/4)), zero=pi,labels=c(90,45,0,-45,-90,-135,180,135),
              rotation='clock', cex=1)
# ticks.circular(circular(seq(0,2*pi,pi/8)), zero=pi/2, rotation='clock', tcl=0.075)

## now plor wt
to_plot<-wt.angle
plot.circular(to_plot,stack = TRUE,pch = 20, sep = sep, start.sep=0.04,shrink = shrink
              ,main='wt',zero=pi/2,cex=cex,axes=FALSE)
rose.diag(to_plot, bins=bins, col="darkgrey", cex=1.5, prop=prop,
          add=TRUE,axes=FALSE,zero=pi/2)
# lines(density.circular(to_plot, bw=kernel_band),zero=pi/2, lwd=2, lty=3)
axis.circular(at=circular(seq(0, 7*pi/4,pi/4)), zero=pi,labels=c(90,45,0,-45,-90,-135,180,135),
              rotation='clock', cex=1)
ticks.circular(circular(seq(0,2*pi,pi/8)), zero=pi/2, rotation='clock', tcl=0.075)

# Test if the distribtuions are uniform, inddeed disturions are skweded
rayleigh.test(wt.angle)
rayleigh.test(tg.angle)

# hay muchos tipos de distriubciones con las que fitear mis datos.
# En estadistica cicular la Von Misses suele ser muy utilizada.
watson.test(wt.angle, dist="vonmises",alpha=0.05)
watson.test(tg.angle, dist="vonmises",alpha=0.05)

# parace que los datos tg no siguen una Von misses, sin embargo,
# puedo comparar distribuciones simplemente con el watson 2 test.
watson.two.test(tg.angle, wt.angle)

# Both distributuions aree indeed diferent
WatsonU2TestRand(tg.angle, wt.angle,NR=9999)

### mean test
cdat1=tg.angle
cdat2=wt.angle
cdat <- c(cdat1, cdat2)
n1 <- length(cdat1) ; n2 <- length(cdat2)
ndat <- c(n1, n2) ;
g <- 2
YgObs <- YgVal(cdat, ndat, g) ;
pchisq(YgObs, g-1, lower.tail=F)

#median test
PgObs <- PgVal(cdat, ndat, g)
pchisq(PgObs, g-1, lower.tail=F)

length(wt.angle)

length(tg.angle)


# plot(tg.approach$mean_angle,1-tg.approach$magnitude, pch=16, xlab="angle", ylab="magnitude")
# plot(wt.approach$mean_angle,1-wt.approach$magnitude, pch=16, xlab="angle", ylab="magnitude")
