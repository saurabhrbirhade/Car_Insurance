from flask import Flask,jsonify,render_template,request
import utils
# from utils import classifier_model
  
app = Flask(__name__)
@ app.route("/")
def home():
    return render_template('home.html')

@ app.route("/predict",methods = ['GET'])
def predict():
    data    = request.args
    policy_tenure = float(data["policy_tenure"])
    age_of_car    = float(data["age_of_car"])
    age_of_policyholder   = float(data["age_of_policyholder"])
    area_cluster    = (data["area_cluster"])
    population_density   = float(data["population_density"])
    make   = float(data["make"])
    segment    = (data["segment"])
    model    = (data["model"])
    fuel_type    = (data["fuel_type"])
    max_torque    = (data["max_torque"])
    max_power    = (data["max_power"])
    engine_type    = (data["engine_type"])
    airbags   = float(data["airbags"])
    is_esc    = (data["is_esc"])
    is_adjustable_steering    = (data["is_adjustable_steering"])
    is_tpms    = (data["is_tpms"])
    is_parking_sensors    = (data["is_parking_sensors"])
    is_parking_camera    = (data["is_parking_camera"])
    rear_brakes_type    = (data["rear_brakes_type"])
    displacement   = float(data["displacement"])
    cylinder   = float(data["cylinder"])
    transmission_type    = (data["transmission_type"])
    gear_box   = float(data["gear_box"])
    steering_type    = (data["steering_type"])
    turning_radius   = float(data["turning_radius"])
    length   = float(data["length"])
    width   = float(data["width"])
    height   = float(data["height"])
    gross_weight   = float(data["gross_weight"])
    is_front_fog_lights    = (data["is_front_fog_lights"])
    is_rear_window_wiper    = (data["is_rear_window_wiper"])
    is_rear_window_washer    = (data["is_rear_window_washer"])
    is_rear_window_defogger    = (data["is_rear_window_defogger"])
    is_brake_assist    = (data["is_brake_assist"])
    is_power_door_locks    = (data["is_power_door_locks"])
    is_central_locking    = (data["is_central_locking"])
    is_power_steering    = (data["is_power_steering"])
    is_driver_seat_height_adjustable    = (data["is_driver_seat_height_adjustable"])
    is_day_night_rear_view_mirror    = (data["is_day_night_rear_view_mirror"])
    is_ecw    = (data["is_ecw"])
    is_speed_alert    = (data["is_speed_alert"])
    ncap_rating   = float(data["ncap_rating"])

    predictor = utils.classifier_model(policy_tenure, age_of_car, age_of_policyholder,
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
       is_ecw, is_speed_alert, ncap_rating)
    prediction = predictor.predict()
    
    if prediction == 1:
        return "Policyholder will claim in upcoming 6 months"
    else:
        return "Policyholder will not claim"

if __name__ == "__main__":
    app.run()