#!/usr/bin/env python
import numpy
import Image
import os

from astropy.io import fits

def jpgToFITS(filename):
	'''Extract the RGB channels and save them as FITS.'''
	#get the image and color information
	image = Image.open(filename)
	image.load()
	try:
		timestamp=image._getexif()[306]
	except:
		timestamp=''
	#306=datetime
	#36867: 'DateTimeOriginal',
	#36868: 'DateTimeDigitized',
	#33434: 'ExposureTime',
	#33437: 'FNumber',
	#PIL.ExifTags.Tags gives all the tag names
	#image.show()
	xsize, ysize = image.size
	r, g, b = image.split()
	rdata = r.getdata() # data is now an array of length ysize\*xsize
	gdata = g.getdata()
	bdata = b.getdata()

	# create numpy arrays
	npr = numpy.reshape(rdata, (ysize, xsize))
	npg = numpy.reshape(gdata, (ysize, xsize))
	npb = numpy.reshape(bdata, (ysize, xsize))

	# write out the fits images, the data numbers are still JUST the RGB
	# scalings; don't use for science
	red = fits.PrimaryHDU(data=npr)
	red.header['TIMESTAMP'] = timestamp # add spurious header info
	red.header['CHANNEL'] = "red"
	red.writeto(os.path.splitext(filename)[0]+'_r.fits',clobber=True)

	green = fits.PrimaryHDU(data=npg)
	green.header['TIMESTAMP'] = timestamp # add spurious header info
	green.header['CHANNEL'] = "green"
	green.writeto(os.path.splitext(filename)[0]+'_g.fits',clobber=True)

	blue = fits.PrimaryHDU(data=npb)
	blue.header['TIMESTAMP'] = timestamp # add spurious header info
	blue.header['CHANNEL'] = "blue"
	blue.writeto(os.path.splitext(filename)[0]+'_b.fits',clobber=True)
	

if __name__=="__main__":
	jpgToFITS(r'E:\work\workSlumping\dusttest\IMG_0347.jpg')