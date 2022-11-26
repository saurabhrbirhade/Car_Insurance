import config
import pickle
import json
import numpy as np
 
class classifier_model():
    def __init__(self,policy_tenure, age_of_car, age_of_policyholder,
       area_cluster, population_density, make, segment, model,
       fuel_type, max_torque, max_power, engine_type, airbags,
       is_esc, is_adjustable_steering, is_tpms, is_parking_sensors,
       is_parking_camera, rear_brakes_type, displacement, cylinder,
       transmission_type, gear_box, steering_type, turning_radius,
       length, width, height, gross_weight, is_front_fog_lights,
       is_rear_window_wiper, is_rear_window_washer,
       is_rear_window_defogger, is_brake_assist, is_power_door_locks,
       is_central_locking, is_power_steering,
       is_driver_seat_height_adjustable, is_day_night_rear_view_mirror,
       is_ecw, is_speed_alert, ncap_rating):
        self.policy_tenure        = policy_tenure
        self.age_of_car           = age_of_car
        self.age_of_policyholder  = age_of_policyholder
        self.area_cluster         = area_cluster
        self.population_density   = population_density
        self.make                 = make
        self.segment              = segment
        self.model                = model
        self.fuel_type            = fuel_type
        self.max_torque           = max_torque
        self.max_power            = max_power
        self.engine_type          = engine_type
        self.airbags              = airbags
        self.is_esc               = is_esc
        self.is_adjustable_steering = is_adjustable_steering
        self.is_tpms               = is_tpms
        self.is_parking_sensors    = is_parking_sensors
        self.is_parking_camera     = is_parking_camera
        self.rear_brakes_type      = rear_brakes_type
        self.displacement          = displacement
        self.cylinder              = cylinder
        self.transmission_type     = transmission_type
        self.gear_box              = gear_box
        self.steering_type         = steering_type
        self.turning_radius        = turning_radius
        self.length                = length
        self.width                 = width
        self.height                = height
        self.gross_weight          = gross_weight
        self.is_front_fog_lights   = is_front_fog_lights
        self.is_rear_window_wiper  = is_rear_window_wiper
        self.is_rear_window_washer = is_rear_window_washer
        self.is_rear_window_defogger = is_rear_window_defogger
        self.is_brake_assist        = is_brake_assist
        self.is_power_door_locks    = is_power_door_locks
        self.is_central_locking     = is_central_locking
        self.is_power_steering      = is_power_steering
        self.is_driver_seat_height_adjustable = is_driver_seat_height_adjustable
        self.is_day_night_rear_view_mirror    = is_day_night_rear_view_mirror
        self.is_ecw                 = is_ecw
        self.is_speed_alert         = is_speed_alert
        self.ncap_rating            = ncap_rating

    def load_data(self):
        with open (config.MODEL_PATH,'rb') as f:
            self.RFmodel = pickle.load(f)
        with open(config.Json_path,'r') as f:
            self.encoded_columns = json.load(f)

    def predict(self):
        self.load_data()
        details = np.zeros(len(self.encoded_columns['col_list']))

        details[0]    = (self.policy_tenure)
        details[1]    = (self.age_of_car)
        details[2]    = (self.age_of_policyholder)
        details[3]    = int(self.area_cluster.replace("C",''))
        details[4]    = (self.population_density)
        details[5]    = (self.make)
        details[6]    = (self.airbags)
        is_esc_val       = self.encoded_columns['Bool_col'][self.is_esc]
        details[7]    = is_esc_val
        is_adj_steer_val = self.encoded_columns['Bool_col'][self.is_adjustable_steering]
        details[8]    = is_adj_steer_val
        is_tpms_val      = self.encoded_columns['Bool_col'][self.is_tpms]
        details[9]    = is_tpms_val
        is_parking_sensors_val    = self.encoded_columns['Bool_col'][self.is_parking_sensors]
        details[10] = is_parking_sensors_val
        is_parking_camera_val    = self.encoded_columns['Bool_col'][self.is_parking_camera]
        details[11] = is_parking_camera_val
        details[12] = (self.displacement)
        details[12] = (self.cylinder)
        transmission_type_val = self.encoded_columns['transmission_type'][self.transmission_type]
        details[13] = transmission_type_val
        details[14] = (self.gear_box)
        steering_type_val = self.encoded_columns['steering_type'][self.steering_type]
        details[15] = steering_type_val
        details[16] = (self.turning_radius)
        details[17] = (self.length)
        details[18] = (self.width)
        details[19] = (self.height)
        details[20] = (self.gross_weight)
        is_front_fog_lights_val    = self.encoded_columns['Bool_col'][self.is_front_fog_lights]
        details[21] = is_front_fog_lights_val
        is_rear_window_wiper_val    = self.encoded_columns['Bool_col'][self.is_rear_window_wiper]
        details[22] = is_rear_window_wiper_val
        is_rear_window_washer_val    = self.encoded_columns['Bool_col'][self.is_rear_window_washer]
        details[23] = is_rear_window_washer_val
        is_rear_window_defogger_val    = self.encoded_columns['Bool_col'][self.is_rear_window_defogger]
        details[24] = is_rear_window_defogger_val
        is_brake_assist_val    = self.encoded_columns['Bool_col'][self.is_brake_assist]
        details[25] = is_brake_assist_val
        is_power_door_locks_val    = self.encoded_columns['Bool_col'][self.is_power_door_locks]
        details[26] = is_power_door_locks_val
        is_central_locking_val    = self.encoded_columns['Bool_col'][self.is_central_locking]
        details[27] = is_central_locking_val
        is_power_steering_val    = self.encoded_columns['Bool_col'][self.is_power_steering]
        details[28] = is_power_steering_val
        is_driver_seat_height_adjustable_val    = self.encoded_columns['Bool_col'][self.is_driver_seat_height_adjustable]
        details[29] = is_driver_seat_height_adjustable_val
        is_day_night_rear_view_mirror_val    = self.encoded_columns['Bool_col'][self.is_day_night_rear_view_mirror]
        details[30] = is_day_night_rear_view_mirror_val
        is_ecw_val    = self.encoded_columns['Bool_col'][self.is_ecw]
        details[31] = is_ecw_val
        is_speed_alert_val    = self.encoded_columns['Bool_col'][self.is_speed_alert]
        details[32] = is_speed_alert_val
        details[33] = (self.ncap_rating)
        segment_index = self.encoded_columns['col_list'].index('segment_'+self.segment)
        details[segment_index] = 1

        model_index = self.encoded_columns['col_list'].index('model_'+self.model)
        details[model_index] = 1

        fuel_type_index = self.encoded_columns['col_list'].index('fuel_type_'+self.fuel_type)
        details[fuel_type_index] = 1

        engine_type_index = self.encoded_columns['col_list'].index('engine_type_'+self.engine_type)
        details[engine_type_index] = 1

        rear_brakes_type_index = self.encoded_columns['col_list'].index('rear_brakes_type_'+self.rear_brakes_type)
        details[rear_brakes_type_index] = 1

        fuel_type_index = self.encoded_columns['col_list'].index('fuel_type_'+self.fuel_type)
        details[fuel_type_index] = 1

        max_tor_list = self.max_torque.split('@')
        max_torque_val = float(max_tor_list[1].replace('rpm','')) / float(max_tor_list[0].replace('Nm',''))
        details[-2] = max_torque_val

        max_power_list = self.max_power.split('@')
        max_power_val = float(max_power_list[1].replace('rpm','')) / float(max_power_list[0].replace('bhp',''))
        details[-1] = max_power_val

        probabilities = self.RFmodel.predict_proba([details])[:,0]
        if probabilities >=0.583:
            return 0
        else:
            return 1


        # prediction = self.RFmodel.predict([details])

        # return prediction