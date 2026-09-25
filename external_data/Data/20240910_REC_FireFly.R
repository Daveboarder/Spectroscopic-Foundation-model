rm (list=ls(all=TRUE))
library(ggplot2)
library(plotly)
library(jsonlite)
library(rhdf5)

setwd("P:/Running_projects/24_0015_ML_Mapping/")

uv_calib_wl <- c(180.00,220.00,230.00,240.00,250.00,260.00,270.00,280.00,290.00,300.00,310.00,320.00,330.00,340.00,350.00,360.00,370.00,380.00,390.00,400.00)
uv_calib_int <- c(4.14924E1,3.1362E1,2.9241E1,2.6387E1,2.3884E1,2.1531E1,1.9062E1,1.6373E1,1.4108E1,1.2303E1,1.098E1,9.5562E0,8.5976E0,7.7735E0,7.0041E0,6.4071E0,5.9195E0,5.6728E0,5.1471E0,5.1181E0)
uv_lamp <- approxfun(x= uv_calib_wl, y = uv_calib_int, method="linear", rule = 1, f = 0, ties = mean)
uv_range <- c(180,400)
vis_calib_wl <- c( 350.00,360.00,	370.00,	380.00,	390.00,	400.00,	420.00,	440.00,	460.00,	480.00,	500.00,	525.00,	550.00,	575.00,	600.00,	650.00,	700.00,	750.00,	800.00,	850.00,	900.00,	950.00,	1000.00,1050.00)
vis_calib_int <- c(1.9862E-1,2.2677E-1,2.6293E-1,2.9921E-1,3.3594E-1,3.7371E-1,5.0574E-1,6.9292E-1,9.3803E-1,1.2383E0,1.5973E0,2.0941E0,2.6431E0,3.261E0,3.9116E0,5.2424E0,6.6042E0,7.8885E0,9.0153E0,1.0003E1,1.0732E1,1.0926E1,1.0511E1,9.65E0)
vis_lamp <- approxfun(x= vis_calib_wl, y = vis_calib_int, method="linear", rule = 1, f = 0, ties = mean)
vis_range <- c(350,900)
valid_calibration <- c(uv_range[1],vis_range[2])

#Deuterium lamp========================================================
Data.list <- list.files(pattern = "*.h5", full.names = F)
data <- h5read(Data.list[1], "/measurements")
wavelengths <- data[[1]][[2]][[1]]
spectra <- as.data.frame(data[[1]][[2]][[2]])
calibSpectraUV <- lowess(spectra[wavelengths>uv_range[1] & wavelengths<uv_range[2],1],f=0.01)$y

calibSpectraUV[which(calibSpectraUV==0)] <- (calibSpectraUV[which(calibSpectraUV==0)+1]+calibSpectraUV[which(calibSpectraUV==0)+2])/2
#----------------------------------------------------------------------
uv_calibration <- approxfun(x=wavelengths[wavelengths>uv_range[1]& wavelengths<uv_range[2]],
                            y=uv_lamp(wavelengths[wavelengths>uv_range[1]& wavelengths<uv_range[2]])/calibSpectraUV,
                            method="linear", rule = 1, f = 0, ties = mean)

plot(wavelengths[wavelengths>uv_range[1]& wavelengths<uv_range[2]],uv_calibration(wavelengths[wavelengths>uv_range[1]& wavelengths<uv_range[2]]), type = "l")

unorm <- function(x){x <- x-min(x)
                    x <- x/max(x)
                    return(x)}

plotD <- data.frame(wavelengths=wavelengths[wavelengths>uv_range[1] & wavelengths<uv_range[2]], spctr = unorm(calibSpectraUV),
                    teorLamp=unorm(uv_lamp(wavelengths[wavelengths>uv_range[1] & wavelengths<uv_range[2]])),
                    ratio=uv_calibration(wavelengths[wavelengths>uv_range[1] & wavelengths<uv_range[2]]))
pl <- ggplot(plotD,aes(wavelengths, spctr, color='Duterium'))+geom_line()+geom_line(aes(wavelengths,teorLamp, color='theoretical'))+
      geom_line(aes(wavelengths,ratio,color='Correction')) + theme_bw() +
      labs(x = "Wavelengths (nm)", y="Intensity (a.u.)", colour="") + 
      theme(axis.title.x=element_text(face="bold",size=15),axis.text.x=element_text(size=15)) + 
      theme(axis.title.y=element_text(face="bold",size=15),axis.text.y=element_text(size=15))
ggplotly(pl)
#Halogen lamp==========================================================
data <- h5read(Data.list[2], "/measurements")
wavelengths <- data[[1]][[2]][[1]]
spectra <- as.data.frame(data[[1]][[2]][[2]])
calibSpectraVIS <- spectra[wavelengths>vis_range[1] & wavelengths<vis_range[2],1]

calibSpectraVIS[which(calibSpectraVIS==0)] <- (calibSpectraVIS[which(calibSpectraVIS==0)-1]+calibSpectraVIS[which(calibSpectraVIS==0)-2])/2
#----------------------------------------------------------------------
vis_calibration <- approxfun(x=wavelengths[wavelengths>vis_range[1]& wavelengths<vis_range[2]],
                             y=vis_lamp(wavelengths[wavelengths>vis_range[1]& wavelengths<vis_range[2]])/calibSpectraVIS,
                             method="linear", rule = 1, f = 0, ties = mean) 
plot(wavelengths[wavelengths>vis_range[1]& wavelengths<vis_range[2]],vis_calibration(wavelengths[wavelengths>vis_range[1]& wavelengths<vis_range[2]]), type = "l")
#======================================================================
uvisNorm<-data.frame(uv=uv_calibration(wavelengths[wavelengths>vis_range[1]& wavelengths<uv_range[2]]),vis=vis_calibration(wavelengths[wavelengths>vis_range[1]& wavelengths<uv_range[2]]))
linFit <- lm(vis~uv,uvisNorm)
#uv kalibrace je o 2.930e-6 nize a 0.1547 krat mensi ... kompenzuji

plot(wavelengths[wavelengths>vis_range[1]& wavelengths<uv_range[2]],vis_calibration(wavelengths[wavelengths>vis_range[1]& wavelengths<uv_range[2]]),type="l")
lines(wavelengths[wavelengths>vis_range[1]& wavelengths<uv_range[2]],linFit$coefficients[[1]]+linFit$coefficients[[2]]*uv_calibration(wavelengths[wavelengths>vis_range[1]& wavelengths<uv_range[2]]),col=2)

plot(wavelengths[wavelengths>vis_range[1]& wavelengths<uv_range[2]],vis_calibration(wavelengths[wavelengths>vis_range[1]& wavelengths<uv_range[2]])/uv_calibration(wavelengths[wavelengths>vis_range[1]& wavelengths<uv_range[2]]))

uvisCalibration <- approxfun(x=c(wavelengths[wavelengths>uv_range[1]& wavelengths<uv_range[2]],wavelengths[wavelengths>vis_range[1]& wavelengths<vis_range[2]]),y=c(linFit$coefficients[[1]]+linFit$coefficients[[2]]*uv_calibration(wavelengths[wavelengths>uv_range[1]& wavelengths<uv_range[2]]),vis_calibration(wavelengths[wavelengths>vis_range[1]& wavelengths<vis_range[2]])),ties=mean,method="linear",rule=1)
plot(wavelengths, uvisCalibration(wavelengths)/max(na.omit(uvisCalibration(wavelengths))),type="l")

plotData <- data.frame(wavelengths=wavelengths, intensity=uvisCalibration(wavelengths))

plT <- ggplot(plotData,aes(wavelengths, intensity)) + geom_line() + theme_bw()
ggplotly(plT)

write.table(plotData, file = "D:/ownCloud/Documents/R-projects/R functions/FireFlyUvisCalibration.txt", sep = "\t", dec = ".")

save("uvisCalibration", file="D:/ownCloud/Documents/R-projects/R functions/FireFlyUvisCalibration.Rdata")

wvl <-as.numeric(wavelengths[wavelengths>uv_range[1]&wavelengths<uv_range[2]])
plotData <- data.frame(x=wvl, y=uv_calibration(wvl)/max(na.omit(uv_calibration(wavelengths))))


#Data for sensitivity
plotSensitivity <- data.frame(x=wvl, y=1/uv_calibration(wvl))

sensitivity <- ggplot(plotSensitivity) + geom_line(aes(x=x, y=y)) + theme_bw() + labs(x = "Wavelength (nm)", y="Sensitivity (-)", colour="", title = "Avantes_ULS4096CL") +
  theme(axis.title.x=element_text(size=15),axis.text.x=element_text(size=15)) + 
  theme(axis.title.y=element_text(size=15),axis.text.y=element_text(size=15)) + theme(plot.title = element_text(size=20))

ggplotly(sensitivity)

plo <- ggplot(plotData) + geom_line(aes(x,y, color = 'black')) + theme_bw() + labs(x = "Wavelength (nm)", y="Sensitivity (-)", colour="")
ggplotly(plo)

max(as.numeric(wavelengths))
min(as.numeric(wavelengths))

res <- c(as.numeric(wavelengths),0) - c(0,as.numeric(wavelengths))
min(res[c(-1,-length(res))])
