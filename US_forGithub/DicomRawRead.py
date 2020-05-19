#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys
import array

import dicom
import pylab
import glob
from skimage.draw import polygon   #pip scikit-image


# RT-structure (http://aapmchallenges.cloudapp.net/forums/3/2/)
def read_structure(structure):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].RefdROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours

def get_mask(contours, slices, imageCT):
    z = [s.ImagePositionPatient[2] for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]
    label = np.zeros_like(imageCT, dtype=np.uint8)
    for con in contours:
        num = int(con['number'])
        print("struct No ",num, con['name'])
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            nodes[(nodes>-0.0001) & (nodes<0.0001)] = 0     #変な値修正
            #print(nodes)
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(nodes[0, 2])
            #print(z_index)
            r = (nodes[:, 1] - pos_r) / spacing_r          #posi norm
            c = (nodes[:, 0] - pos_c) / spacing_c          #posi norm
            rr, cc = polygon(r, c)
            label[rr, cc, z_index] = num
            colors = tuple(np.array([con['color'] for con in contours]) / 255.0)  #ピクセル値でstructure No.を表示
    return label, colors, num

## --- Loading "One" Simple Rawdata file (not directory) ---
def load_One_Simple_Raw_image(train_data_image_filename, colsize, rowsize, bb, rgb):
    openfilename = train_data_image_filename
    f = open(openfilename, 'rb')
    if bb == "int16":
        bdata = array.array('h')  # 'h': signed short (2 byte = 16 bit)
    elif bb == "uint8":
        bdata = array.array('B')  # 'B': unsigned char (1 byte = 8 bit)
    elif bb == "float64":
        bdata = array.array('d')  # 'd': double (8 byte = 64 bit)
    elif bb == "float32":
        bdata = array.array('f')  # 'f': float (4 byte = 32 bit)
    bdata.fromfile(f, colsize*rowsize*rgb)
    #if sys.byteorder == 'little':
    #    bdata.byteswap()
#    print(openfilename, len(bdata))
    f.close()
    re_image = np.frombuffer(bdata,dtype=bb)
    return re_image

def load_One_Simple_Raw_image_3D(train_data_image_filename, colsize, rowsize, height, bb):
    openfilename = train_data_image_filename
    f = open(openfilename, 'rb')
    if bb == "int16":
        bdata = array.array('h')  # 'h': signed short (2 byte = 16 bit)
    elif bb == "int8":
        bdata = array.array('B')  # 'd': unsigned char (1 byte = 8 bit)
    elif bb == "float64":
        bdata = array.array('d')  # 'd': double (8 byte = 64 bit)
    elif bb == "float32":
        bdata = array.array('f')  # 'f': float (4 byte = 32 bit)
    bdata.fromfile(f, colsize*rowsize*height)
    #if sys.byteorder == 'little':
    #    bdata.byteswap()
    #print(openfilename, len(bdata))
    f.close()
    re_image = np.frombuffer(bdata,dtype=bb)
    return re_image

## --- Loading "One" Rawdata file (tiff) in US (not directory) ---
def load_One_US_Raw_image(openfilename, colsize, rowsize, rgb):
    f = open(openfilename, 'rb')
    tiff1 = f.read(2)# 4D 4D # Big: 4D, Little: 49
    tiff2 = f.read(2)# 00 2A means tiff format
    tiff3 = f.read(4)
    bdata = f.read(colsize*rowsize*rgb)
    f.close()
    image = np.asarray(bytearray(bdata), dtype=np.uint8) # bytearray => 0 < x < 255
    return image


## --- Loading Simple Rawdata file ---
def load_Simple_Raw_image(train_data_image_path, colsize, rowsize, rgb):
    dcm_list_h=[]
    file_count = 0
    for name in os.listdir(train_data_image_path):
        _, ext = os.path.splitext(name)
        if (ext == '.dcm'):
            dcm_list_h.append(name)
            file_count += 1
        #print(file_count, name)
        elif (ext == '.raw'):
            dcm_list_h.append(name)
            file_count += 1
    #print(file_count, name)

    print("No. of images",file_count)
    dcm_list_h = sorted(dcm_list_h)
    for i in range(file_count):
        openfilename = train_data_image_path + dcm_list_h[i]
        f = open(openfilename, 'rb')
        bdata = array.array('h')  # 'h': signed short (2 byte = 16 bit)
        bdata.fromfile(f, colsize*rowsize)
        #if sys.byteorder == 'little':
        #    bdata.byteswap()
        print(i, openfilename, len(bdata))
        f.close()
        re_image = np.frombuffer(bdata,dtype="int16")
    
    #norm_image = np.asarray(bdata, dtype=np.uint8)
    #f = open('test.raw', 'wb')
    #    f.write(bdata)
    #    f.close()
    #a = struct.unpack("d", f.read(8*512*512))
    #    for i in range(yst,yen):
    #            for j in range(xst,xen):
    #    a = struct.unpack("d", f.read(8))
    #    print(a)
    #    im[i,j] = a
    #    f.close
    #    for i in range(3):
    #        plt.subplot(1, 3, i+1)
    #        plt.imshow(im[:,:,i])
    #    plt.show()
    return re_image

## --- Loading Rawdata file (tiff) in US ---
def load_US_Raw_image(train_data_image_path, colsize, rowsize, rgb):
    dcm_list_h=[]
    file_count = 0
    for name in os.listdir(train_data_image_path):
        _, ext = os.path.splitext(name)
        if (ext == '.dcm'):
            dcm_list_h.append(name)
            file_count += 1
            #print(file_count, name)
        elif (ext == '.raw'):
            dcm_list_h.append(name)
            file_count += 1
            #print(file_count, name)

    print("No. of images",file_count)
    dcm_list_h = sorted(dcm_list_h)
    for i in range(file_count):
        openfilename = train_data_image_path + dcm_list_h[i]
        f = open(openfilename, 'rb')
        tiff1 = f.read(2)# 4D 4D # Big: 4D, Little: 49
        tiff2 = f.read(2)# 00 2A means tiff format
        #tiff3 = array.array('I')  # 'h': signed short (2 byte = 16 bit)
        #tiff3.fromfile(f, 1)
        tiff3 = f.read(4)

        '''
        # header 1
        ssiz = 100
        rsize = int(ssiz*ssiz*3 + 8)
        rsize_s = struct.pack('>1I',rsize)
        cdata3 = tiff1 + tiff2 + rsize_s
        fname = "header_tiff_%d_%d_1.raw" % (ssiz,ssiz)
        f2 = open(fname,'wb')
        f2.write(cdata3)
        f2.close()
        sys.exit()
        '''
        '''
        #---- test ----
        print(tiff3)
        rsize = struct.unpack('>1I',tiff3)
        #rsize_s = hex(rsize[0])
        #moji = str(rsize[0])
        #print(moji)
        #suuchi = int(moji)
        #print(suuchi)
        #print(struct.pack(">1I",suuchi))
        print(struct.pack(">1I",rsize[0]))
        sys.exit()
        '''
        bdata = f.read(colsize*rowsize*rgb)
        #        print(i, openfilename, len(bdata))

        '''
        #---- header 2 ----
        ssiz = 100
        cwid = struct.pack(">1h",ssiz)
        cwid1 = struct.pack(">1I",ssiz*ssiz*3+182)
        cwid2 = struct.pack(">1I",ssiz*ssiz*3)
        cwid3 = struct.pack(">1I",ssiz*ssiz*3+188)
        cwid4 = struct.pack(">1I",ssiz*ssiz*3+194)
        cwid5 = struct.pack(">1I",ssiz*ssiz*3+200)
        #print(cwid)
        cdata1 = f.read(10)
        cdata = f.read(2)
        cdata2 = f.read(10)
        cdata = f.read(2)
        cdata3 = f.read(10)
        cdata = f.read(4)
        cdata4 = f.read(68)
        cdata = f.read(2)
        cdata5 = f.read(10)
        cdata = f.read(4)
        cdata6 = f.read(8)
        cdata = f.read(4)
        cdata7 = f.read(8)
        cdata = f.read(4)
        cdata8 = f.read(20)
        cdata = f.read(4)
        cdata9 = f.read(32)
        cdata = cdata1 + cwid + cdata2 + cwid + cdata3 + cwid1 + cdata4 + cwid + cdata5 + cwid2 + cdata6 + cwid3 + cdata7 + cwid4 + cdata8 + cwid5 + cdata9
        headname = 'header_tiff_%d_%d_2.raw' % (ssiz,ssiz)
        f2 = open(headname,'wb')
        f2.write(cdata)
        f2.close()
        sys.exit()
        '''
        
        f.close()
        image = np.asarray(bytearray(bdata), dtype=np.uint8) # bytearray => 0 < x < 255
        #value = image[(rowsize-64)*rgb:]                                       # what is this?
        #re_image = np.append(value, np.zeros((rowsize-64)*rgb))                # what is this?
        re_image = image
        if i == 0:
            images = re_image
        else:
            images = np.append(images, re_image)
#    im = np.reshape(re_image,(colsize,rowsize,-1))

#norm_image = np.asarray(bdata, dtype=np.uint8)
#f = open('test.raw', 'wb')
#    f.write(bdata)
#    f.close()
    #a = struct.unpack("d", f.read(8*512*512))
    #    for i in range(yst,yen):
#            for j in range(xst,xen):
#    a = struct.unpack("d", f.read(8))
#    print(a)
#    im[i,j] = a
#    f.close
#    for i in range(3):
#        plt.subplot(1, 3, i+1)
#        plt.imshow(im[:,:,i])
#    plt.show()
#print(images.dtype)
    return re_image, images, file_count

def load_DICOM_image(train_data_image_path):
    
    dcm_list_h=[]
    file_count = 0
    for name in os.listdir(train_data_image_path):
        _, ext = os.path.splitext(name)
        if ext == '.dcm':
            dcm_list_h.append(name)
            file_count += 1

    print("No. of images",file_count)


    ref_dicom = dicom.read_file(train_data_image_path + dcm_list_h[0])     
    dcm_rows = ref_dicom.Rows #dcm[0x0028, 0x0010]
    dcm_cols = ref_dicom.Columns #dcm[0x0028, 0x0011]
    dcm_thickness = ref_dicom.SliceThickness
    dcm_Bits = ref_dicom.BitsAllocated
    dcm_PixelSpacing =ref_dicom.PixelSpacing
    dcm_position=ref_dicom.ImagePositionPatient 
    #print("ref_image")
    #print(dcm_list_h[0])
    #print(dcm_rows, dcm_cols, dcm_thickness ,dcm_PixelSpacing[0], dcm_Bits)
    #print(dcm_position[0], dcm_position[1], dcm_position[2])
    
    slices = [dicom.read_file(train_data_image_path + dcm_file) for dcm_file in dcm_list_h]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    #dcm_pisiton=x.ImagePositionPatient[2]
    z_position = []
    for s in slices:
        z_position.append(s.ImagePositionPatient[2])
    z_position.sort
    #print("maxposition=", max(z_position))
    #print("minposition=",min(z_position))
    #print(z_position)
    '''
    biggest= np.amax(image)
    smallest=np.amin(image)
    print("image max min", biggest, smallest)
    '''
    #print(image)

    ###for keras
    norm_image = np.asarray(image)
    
    #im = norm_image[:,:,1]
    #plt.imshow(im)
    #plt.show()

    ###normalize 0-1#####
    biggest= np.amax(norm_image)
    smallest=np.amin(norm_image)
    #print("image max min", biggest, smallest)
    norm_image = norm_image.astype('float32')
    norm_image = (norm_image -  smallest) / (biggest - smallest) # CT-value とりあえず。他のCTと正規化揃える？
    biggest= np.amax(norm_image)
    smallest=np.amin(norm_image)
    #print("after norm.", biggest, smallest)
    #######
    
    print (norm_image.shape)
    return norm_image, image, dcm_list_h, slices, file_count, dcm_rows, dcm_cols, dcm_thickness, dcm_PixelSpacing, z_position

# struct画像を読み込む関数
def load_Struct(train_data_struct_path, slices, imageCT, file_count,dcm_rows, dcm_cols):

    dcm_list_S = []
    for name in os.listdir(train_data_struct_path ):
        _, ext = os.path.splitext(name)
        if ext == '.dcm':
            dcm_list_S.append(name)
            structure = dicom.read_file(train_data_struct_path + dcm_list_S[0])
            contours = read_structure(structure)

    image, colors, num = get_mask(contours, slices, imageCT)
 
    ###for keras
    norm_image = np.asarray(image)
    
    ###normalize 0-1#####
    biggest= np.amax(norm_image)
    smallest=np.amin(norm_image)
    print("structure max min", biggest, smallest)
    norm_image = norm_image.astype('float32')
    norm_image = norm_image / biggest
    biggest= np.amax(norm_image)
    smallest=np.amin(norm_image)
    print("after norm.", biggest, smallest)
    #######

    #####################test 0 or 1 mask image
    norm_image = np.ceil(norm_image)
    biggest= np.amax(norm_image)
    smallest=np.amin(norm_image)
    print("after norm.", biggest, smallest)
    ######################

    
    #norm_image = norm_image.reshape(file_count, dcm_rows, dcm_cols, 1) #atode
    print (norm_image.shape)

    return norm_image, image, colors, num


if __name__ == '__main__':
    train_data_image_path = "/mnt/nfs_S65/USdata/EFdata_2019_output/EF10/EF10-001_0001/"
    colsize = 1016
    rowsize = 708
    rgb = 3
    load_US_Raw_image(train_data_image_path, colsize, rowsize, rgb)
    """
    # 3断面表示 image
    norm_image, image, file_names, slices, file_count,dcm_rows, dcm_cols, dcm_thickness, dcm_PixelSpacing = load_CT('./DicomData' + os.sep + 'TestDicomBreast/')
    norm_label, label, colors, num = load_Struct('./DicomData/TestDicomBreast' + os.sep + 'struct/', slices, image, file_count, dcm_rows, dcm_cols)

    aspect_cor = dcm_PixelSpacing[0] / dcm_thickness
    aspect_sag = dcm_PixelSpacing[0] / dcm_thickness
    aspect_ax = dcm_PixelSpacing[0] / dcm_PixelSpacing[1]
    print(image.shape)
    print(label.shape)
    

    plt.subplot(2, 3, 1)
    #plt.imshow(img_array[:, :, 40], cmap=pylab.cm.bone, aspect=aspect_ax)    #ax
    plt.imshow(image[:, :, 40], cmap=pylab.cm.bone, aspect=aspect_ax)    #ax
    plt.contour(label[:, :, 40], levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors=colors)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    #plt.imshow(img_array[256, :, :], cmap=pylab.cm.bone, aspect=aspect_cor)   #cor
    plt.imshow(image[256, :, :], cmap=pylab.cm.bone, aspect=aspect_cor) 
    plt.contour(label[256, :, ], levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors=colors)
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(image[:, 256, :], cmap=pylab.cm.bone, aspect=aspect_sag)   #sag
    plt.contour(label[:, 256, :], levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors=colors)
    plt.axis('off')
    
    #mask 表示
    plt.subplot(2, 3, 4)
    plt.imshow(label[:, :, 40],  aspect=aspect_ax, vmax=num)    #axi
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(label[256, :, :], aspect=aspect_cor, vmax=num)   #sag
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(label[:, 256, :], aspect=aspect_sag, vmax=num)   #cor
    plt.colorbar()
    plt.axis('off')
    
    plt.show()
    plt.clf()


    plt.subplot(2, 1, 1)
    plt.imshow(norm_image[:, :, 40], cmap=pylab.cm.bone, aspect=aspect_ax)    #ax
    plt.subplot(2, 1, 2)
    plt.imshow(norm_label[:, :, 40],  aspect=aspect_ax)    #ax
    plt.show()
    plt.clf()
    """
