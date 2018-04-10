#!/usr/bin/pvpython
import subprocess, os, math
from paraview.simple import *
import argparse

def makeVideo(pvdPath, statePath, videoPath, dataMin, dataMax):
    print("./%s" % os.path.relpath(pvdPath))

    # paraview tends to get slow after a while
    if not hasattr(makeVideo, "calls"):
        makeVideo.calls = 0

    makeVideo.calls += 1
    if (makeVideo.calls % 40 == 0):
        pxm = servermanager.ProxyManager()
        pxm.UnRegisterProxies()
        del pxm
        Disconnect()
        Connect()

    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'PVD Reader'
    pvd = PVDReader(FileName = pvdPath)

    # get animation scene
    animationScene1 = GetAnimationScene()

    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    renderView1.ViewSize = [1280, 720]

    #help(renderView1)

    DataSliceFile = paraview.servermanager.Fetch(pvd)
    dataName = DataSliceFile.GetPointData().GetArrayName(0)
    vectorname = dataName # different for e.g. u_h, but not for estimate

    print("Array Name = %s" % dataName)

    # get color transfer function/color map for 'uh'
    uhLUT = GetColorTransferFunction(dataName)

    # show data in view
    pvdDisplay = Show(pvd, renderView1)
    # trace defaults for the display properties.
    pvdDisplay.ColorArrayName = ['POINTS', vectorname]
    pvdDisplay.LookupTable = uhLUT
    pvdDisplay.OSPRayScaleArray = vectorname
    pvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    pvdDisplay.GlyphType = 'Arrow'

    # pvdDisplay.ScalarOpacityUnitDistance = 0.08838834764831846
    # pvdDisplay.SetScaleArray = ['POINTS', vectorname]
    # pvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    # pvdDisplay.OpacityArray = ['POINTS', vectorname]
    # pvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'

    # show color bar/color legend
    pvdDisplay.SetScalarBarVisibility(renderView1, True)

    # get opacity transfer function/opacity map for 'uh'
    uhPWF = GetOpacityTransferFunction(dataName)

    #change interaction mode for render view
    renderView1.InteractionMode = '3D'

    # current camera placement for renderView1
    renderView1.CameraPosition = [-10.506489304234304, -3.6446723986339915, 5.890816961048351]
    renderView1.CameraFocalPoint = [-0.020474488549963532, 0.2810726692100919, 0.6766885441215358]
    renderView1.CameraViewUp = [0.4392887049020663, 0.019511778050979575, 0.8981340235525846]
    renderView1.CameraParallelScale = 0.747010253122867

    # Rescale transfer function
    uhLUT.RescaleTransferFunction(0, 10)

    # Rescale transfer function
    uhPWF.RescaleTransferFunction(0, 10)

    if math.isnan(dataMin) or math.isnan(dataMax):
        print("Finding out min and max values")

        amin = 1e300
        amax = -1e300
        tsteps = pvd.TimestepValues

        for i in range(len(tsteps)):
            renderView1.ViewTime = tsteps[i];
            Render()

            DataSliceFile = paraview.servermanager.Fetch(pvd)
            myrange = DataSliceFile.GetPointData().GetArray(0).GetRange()
            if amin > myrange[0]: amin = myrange[0]
            if amax < myrange[1]: amax = myrange[1]

        if math.isnan(dataMin): dataMin = amin
        if math.isnan(dataMax): dataMax = amax

    print("min value = %f, max value = %f" % (dataMin, dataMax))

    # Rescale transfer function
    uhLUT.RescaleTransferFunction(dataMin, dataMax)

    # Rescale transfer function
    uhPWF.RescaleTransferFunction(dataMin, dataMax)

    # create a new 'Warp By Scalar'
    warpByScalar1 = WarpByScalar(Input=pvd)
    warpByScalar1.Scalars = ['POINTS', vectorname]

    # show data in view
    warpByScalar1Display = Show(warpByScalar1, renderView1)
    # trace defaults for the display properties.
    warpByScalar1Display.ColorArrayName = ['POINTS', vectorname]
    warpByScalar1Display.LookupTable = uhLUT
    warpByScalar1Display.OSPRayScaleArray = vectorname
    warpByScalar1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    warpByScalar1Display.GlyphType = 'Arrow'

    warpByScalar1Display.ScalarOpacityUnitDistance = 0.0933762816403584
    #warpByScalar1Display.SetScaleArray = ['POINTS', vectorname]
    #warpByScalar1Display.ScaleTransferFunction = 'PiecewiseFunction'
    #warpByScalar1Display.OpacityArray = ['POINTS', vectorname]
    #warpByScalar1Display.OpacityTransferFunction = 'PiecewiseFunction'

    # hide data in view
    Hide(pvd, renderView1)

    # show color bar/color legend
    warpByScalar1Display.SetScalarBarVisibility(renderView1, True)

    # Properties modified on warpByScalar1
    warpByScalar1.ScaleFactor = 3/max(abs(dataMax), abs(dataMin))

    # create a new 'Annotate Time Filter'
    annotateTimeFilter1 = AnnotateTimeFilter(Input=pvd)

    # Properties modified on annotateTimeFilter1
    annotateTimeFilter1.Format = 't = %4.3f'

    # show data in view
    annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1)

    # Properties modified on annotateTimeFilter1Display
    #annotateTimeFilter1Display.Position = [0.755236, 0.68301]
    annotateTimeFilter1Display.WindowLocation = 'UpperRightCorner'
    annotateTimeFilter1Display.FontSize = 15

    # Properties modified on animationScene1
    animationScene1.PlayMode = 'Real Time'

    # Properties modified on animationScene1
    animationScene1.Duration = 6


    if videoPath != "":
      print("Writing %s" % videoPath)
      WriteAnimation(videoPath, Magnification=1, FrameRate=30.0, Quality=2)

    if statePath != "":
      print("Writing %s" % statePath)
      SaveState(statePath)

    Hide(warpByScalar1)
    Hide(annotateTimeFilter1)
    warpByScalar1Display.SetScalarBarVisibility(renderView1, False)
    pvdDisplay.SetScalarBarVisibility(renderView1, False)
    Delete(warpByScalar1)
    Delete(annotateTimeFilter1)
    Delete(pvd)

def main():
    parser = argparse.ArgumentParser(prog='video2d.py', description='create videos and pvsm files from pvd files with 2D data')
    parser.add_argument('inputfile', metavar='path', type=str, nargs=1,
                        help='path to pvd to process')
    parser.add_argument('--no-video', dest='video', action='store_false',
                        default=True,
                        help='do not create .ogv video')
    parser.add_argument('--no-pvsm', dest='pvsm', action='store_false',
                        default=True,
                        help='do not create .pvsm state file')
    parser.add_argument('--min', dest='min', type=float,
                        default=float('nan'),
                        help='fixed min scaling')
    parser.add_argument('--max', dest='max', type=float,
                        default=float('nan'),
                        help='fixed max scaling')

    args = parser.parse_args()

    inputfile = os.path.abspath(args.inputfile[0]);
    directory = os.path.dirname(inputfile)
    basename = os.path.splitext(os.path.basename(inputfile))[0]

    pvsmfile = ""
    if args.pvsm: pvsmfile = os.path.join(directory, basename+".pvsm")

    ogvfile = ""
    if args.video: ogvfile = os.path.join(directory, basename+".ogv")

    if inputfile != "": makeVideo(inputfile, pvsmfile, ogvfile, args.min, args.max);

if __name__ == "__main__":
   main()
