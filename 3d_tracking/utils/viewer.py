import cv2
import numpy as np

from .utils.object import Object

def point_cloud_2_birdseye(points,
                           bboxes,
                           #ids,
                           res=0.1,
                           side_range=(-10., 10.),  # left-most to right-most
                           fwd_range = (-10., 10.), # back-most to forward-most
                           height_range=(-2., 2.),  # bottom-most to upper-most
                           ):
    palette = {
        0: (0, 255, 0),   # Clase 0: verde, coche
        1: (255, 0, 0),   # Clase 1: azul, peatón
        2: (0, 255, 255),   # Clase 2: amarillo, ciclista
        # Agrega más colores para más clases si es necesario
    }

    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1])) # mascara para filtrar puntos en el rango de distancia hacia adelante
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0])) # mascara para filtrar puntos en el rango de distancia hacia los lados
    filter = np.logical_and(f_filt, s_filt) # mascara para filtrar puntos en el rango de distancia hacia adelante y hacia los lados
    indices = np.argwhere(filter).flatten() # índices de los puntos que cumplen con la máscara

    # puntos que cumplen con la máscara
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    # Tiene que poner x e y como si fuuera una imagen, osea con setido contrario
    x_img = (-y_points / res).astype(np.int32)  # x axis is -x in lidar_f_raw and -y in lidar_f
    y_img = (-x_points / res).astype(np.int32)  # y axis is y in lidar_f_raw and -x in lidar_f

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    # suma o resta un el valor maximo o minimo para que el mínimo sea 0, como en una imagen normal
    x_img -= int(np.floor(side_range[0] / res)) # resta porque side_range[0] es negativo
    y_img += int(np.ceil(fwd_range[1] / res)) 

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = (((pixel_values - height_range[0]) / float(height_range[1] - height_range[0])) * 255).astype(np.uint8)

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res) # ancho de la imagen
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res) # alto de la imagen
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    im_color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if bboxes is not None:
        for type_data in ['Car', 'Pedestrian', 'Bicycle']:
            for cam in bboxes[type_data]:
            
                if bboxes[type_data][cam] != []:
                    bbox = np.array(bboxes[type_data][cam]['boxes'])
                    labels = bboxes[type_data][cam]['labels']
                    if len(bbox)>0:

                        # Dibujar cuadros delimitadores en la imagen
                        x_bb_centros = bbox[:, 0]
                        y_bb_centros = bbox[:, 1]

                        # cambiar el orden de las coordenadas para que coincidan con el orden de las coordenadas de la imagen
                        x_bb_c_img = (-y_bb_centros / res)
                        y_bb_c_img = (-x_bb_centros / res)

                        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
                        x_bb_c_img -= int(np.floor(side_range[0] / res)) # resta porque side_range[0] es negativo
                        y_bb_c_img += int(np.ceil(fwd_range[1] / res)) 

                        # Calcular los vértices de las cajas rotadas
                        centers = [(x_bb_c_img[i], y_bb_c_img[i]) for i in range(len(y_bb_c_img))]  # Centros de las cajas
                        sizes = [((bbox[i, 5]/res), (bbox[i, 4]/res)) for i in range(len(bbox))]  # Tamaños de las cajas
                        angles = -bbox[:, 3]* (180 / np.pi)  # Ángulos de rotación de las cajas
                        
                        
                        # Calcular los vértices de la caja rotada
                        for center, size, angle, label in zip(centers, sizes, angles, labels):
                            box = cv2.RotatedRect(center=center, size=size, angle=angle)
                            box_vertices = cv2.boxPoints(box)
                            box = np.int0(box_vertices)
                            # Dibujar la caja en la imagen
                            color = palette.get(0, (255, 255, 255))  # Blanco si la clase no tiene un color asignado
                            cv2.drawContours(im_color, [box], 0, color, 1)
                            cv2.putText(im_color, f"{label}", (box[0][0], box[0][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                
      
    return im_color