

class NebolabSetup():
    # TURTLEBOT PARAMETER
    #-----------------------------------------------------------------
    TB_L_SI2UNI = 0.06 # in meter
    TB_OFFSET_CENTER_WHEEL = 0.04 # in meter
    
    
    # PARAMETER AND FUNCTIONS FOR CONVERSION BETWEEN PIXEL AND METER
    #-----------------------------------------------------------------
    SCALE_M2PXL = 363.33
    PXL_OFFSET = [-25,0] # for Left camera in pixel
    #offset = [20,0] # for Right camera in pixel
    # origin is midpoint of image (1920 * 1080) with some shift
    PXL_ORIGIN = [960+PXL_OFFSET[0], 540+PXL_OFFSET[1]]

    half_width = 1920 / (2 * SCALE_M2PXL)
    half_height = 1080 / (2 * SCALE_M2PXL)
    FIELD_X = [-half_width, half_width]
    FIELD_Y = [-half_height, half_height]
    
    # IMPORTANT: THE COMPUTATION BELOW REPRESENT THE TRANSFORMATION MATRIX 
    # FROM CAMERA (after flipped) TO FIELD, AND VICE-VERSA
    # - positive y camera is towards door, positive x camera is towards whiteboard
    # - positive y field is towards curtain (opposite of door)
    #       positive x field the same with positive x camera
    # ----------------------------------------------------------------------------
    # Function pixel to meter and vice-versa
    @staticmethod
    def pos_m2pxl(x, y):
        return NebolabSetup.PXL_ORIGIN[0] + int( x * NebolabSetup.SCALE_M2PXL), NebolabSetup.PXL_ORIGIN[1] - int( y * NebolabSetup.SCALE_M2PXL)
    @staticmethod
    def pos_pxl2m(px, py): 
        return float((px - NebolabSetup.PXL_ORIGIN[0])/NebolabSetup.SCALE_M2PXL), float(-(py - NebolabSetup.PXL_ORIGIN[1])/NebolabSetup.SCALE_M2PXL)
