#!/usr/bin/python
# -*- coding:utf-8 -*-


"""
/*******************************************************************************
Copyright (c) 1983-2021 Advantech Co., Ltd.
********************************************************************************
THIS IS AN UNPUBLISHED WORK CONTAINING CONFIDENTIAL AND PROPRIETARY INFORMATION
WHICH IS THE PROPERTY OF ADVANTECH CORP., ANY DISCLOSURE, USE, OR REPRODUCTION,
WITHOUT WRITTEN AUTHORIZATION FROM ADVANTECH CORP., IS STRICTLY PROHIBITED.

================================================================================
REVISION HISTORY
--------------------------------------------------------------------------------
$Log:  $

--------------------------------------------------------------------------------
$NoKeywords:  $
*/

/******************************************************************************
*
* Windows Example:
*    StaticDO.py
*
* Example Category:
*    DIO
*
* Description:
*    This example demonstrates how to use Static DO function.
*
* Instructions for Running:
*    1. Set the 'deviceDescription' for opening the device.
*    2. Set the 'profilePath' to save the profile path of being initialized device.
*    3. Set the 'startPort'as the first port for Do .
*    4. Set the 'portCount'to decide how many sequential ports to operate Do.
*
* I/O Connections Overview:
*    Please refer to your hardware reference manual.
*
******************************************************************************/
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import random
import time
from Automation.BDaq import *
from Automation.BDaq.InstantDoCtrl import InstantDoCtrl
from Automation.BDaq.BDaqApi import AdxEnumToString, BioFailed

deviceDescription = "USB-5862,BID#0"
profilePath = u"C:\Advantech\DAQNavi\Examples\profile\DemoDevice.xml"
startPort = 0
portCount = 1

def AdvInstantDO():
    ret = ErrorCode.Success

    # Step 1: Create a instantDoCtrl for DO function.
    # Select a device by device number or device description and specify the access mode.
    # In this example we use ModeWrite mode so that we can fully control the device,
    # including configuring, sampling, etc.
    instantDoCtrl = InstantDoCtrl(deviceDescription)
    for _ in range(7):
        instantDoCtrl.loadProfile = profilePath
       
        # Step 2: Write DO ports
        dataBuffer = [0] * portCount
        for i in range(startPort, portCount + startPort):
            #inputVal is 255 for all On and 0 for all off 
            inputVal= 255 
            
            dataBuffer[i-startPort] = inputVal
        time.sleep(2) # 2 second delay
        ret = instantDoCtrl.writeAny(startPort, portCount, dataBuffer) # writes the output on the I/O board
        if BioFailed(ret):
            break
        print("DO output completed!")

    print("Port count supported:", instantDoCtrl.features.portCount)

    # Step 3: Close device and release any allocated resource.
    instantDoCtrl.dispose() 

    # If something wrong in this execution, print the error code on screen for tracking.
    if BioFailed(ret):
        enumStr = AdxEnumToString("ErrorCode", ret.value, 256)
        print("Some error occurred. And the last error code is %#x. [%s]" % (ret.value, enumStr))

    return 0


if __name__ == "__main__":
    AdvInstantDO()
