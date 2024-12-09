# models/vehicle_model.py

import numpy as np

class VehicleModel:
    def __init__(self, config, scenario, average_speed, total_travel_time,
                 fuel_efficiency_electric_samples, fuel_efficiency_gasoline_samples,
                 computing_power_samples, carbon_intensity_electric_samples,
                 carbon_intensity_gasoline_samples, efficiency_a_samples, error_term_samples):
        """
        VehicleModel 클래스 초기화
        Parameters:
        - config: 설정 딕셔너리
        - scenario: 시나리오 딕셔너리
        - average_speed: 평균 속도 (km/h)
        - total_travel_time: 총 운행 시간 (시간)
        - fuel_efficiency_electric_samples: 전기 연비 샘플 (numpy 배열)
        - fuel_efficiency_gasoline_samples: 휘발유 연비 샘플 (numpy 배열)
        - computing_power_samples: 컴퓨팅 파워 샘플 (numpy 배열)
        - carbon_intensity_electric_samples: 전기 탄소 배출 계수 샘플 (numpy 배열)
        - carbon_intensity_gasoline_samples: 휘발유 탄소 배출 계수 샘플 (numpy 배열)
        - efficiency_a_samples: 효율성 계수 a 샘플 (numpy 배열)
        - error_term_samples: 오차항 e 샘플 (numpy 배열)
        """
        self.config = config
        self.scenario = scenario
        self.average_speed = average_speed
        self.total_travel_time = total_travel_time

        # NumPy 배열 형태의 샘플된 파라미터
        self.fuel_efficiency_electric = fuel_efficiency_electric_samples
        self.fuel_efficiency_gasoline = fuel_efficiency_gasoline_samples
        self.computing_power = computing_power_samples
        self.carbon_intensity_electric = carbon_intensity_electric_samples
        self.carbon_intensity_gasoline = carbon_intensity_gasoline_samples
        self.efficiency_a = efficiency_a_samples
        self.error_term = error_term_samples

    def calculate_total_electric_energy_and_emissions(self):
        """
        전기 차량 및 자율주행차의 에너지 소비량 및 탄소 배출량 계산 (벡터화)
        Returns:
        - total_electric_energy: 총 전기 에너지 소비량 (kWh)
        - total_carbon_electric: 총 전기 탄소 배출량 (kg CO₂)
        """
        config = self.config
        scenario = self.scenario
        average_speed = self.average_speed
        total_travel_time = self.total_travel_time

        # 운행 시간 계산
        T_av = total_travel_time * scenario['adoption_rate']
        T_non_av = total_travel_time - T_av
        vehicle_ratios = scenario['vehicle_ratios']
        T_electric = T_non_av * vehicle_ratios.get('electric', 0)

        # 전기차 에너지 소비량
        E_electric = (average_speed * T_electric) / self.fuel_efficiency_electric  # 샘플링된 전기 연비 사용
        # energy_reduction을 시나리오에서 직접 가져옴
        clustering_reduction = self.scenario['energy_reduction']['clustering']
        routing_reduction = self.scenario['energy_reduction']['optimal_routing']
        energy_reduction = clustering_reduction + routing_reduction
        energy_reduction = min(energy_reduction, 
                               self.config['general']['energy_reduction']['clustering'] +
                               self.config['general']['energy_reduction']['optimal_routing'])

        # 자율주행차 주행 에너지
        E_drive_av = ((average_speed * T_av) / self.fuel_efficiency_electric) * (1 - energy_reduction)
        E_drive_av = self.efficiency_a * E_drive_av + self.error_term

        # 자율주행차 추가 에너지 소비량
        P_compute = self.computing_power
        P_v2x = config['vehicles']['v2x_power']
        P_data = config['vehicles']['data_center_power']
        E_compute = T_av * P_compute
        E_v2x = T_av * P_v2x
        E_data_center = T_av * P_data
        E_additional_av = self.efficiency_a * (E_compute + E_v2x + E_data_center) + self.error_term

        # RSU 에너지 소비량 계산
        E_RSU = self.calculate_RSU_energy()
        self.E_RSU = E_RSU

        # 총 전기 에너지 소비량 및 탄소 배출량
        total_electric_energy = E_electric + E_drive_av + E_additional_av + E_RSU
        total_electric_carbon = total_electric_energy * self.carbon_intensity_electric  # 샘플링된 탄소 배출 계수 사용

        # 각 에너지 구성 요소를 속성으로 저장
        self.E_electric = E_electric
        self.E_drive_av = E_drive_av
        self.E_compute = E_compute
        self.E_v2x = E_v2x
        self.E_data_center = E_data_center
        # self.E_RSU = E_RSU  # 만약 E_RSU를 계산했다면 추가

        return total_electric_energy, total_electric_carbon

    def calculate_RSU_energy(self):
        """
        RSU의 연간 에너지 소비량을 계산하는 함수
        """
        rsu_power_kw_per_unit = self.config['rsu']['power_kw_per_unit']
        rsu_number_of_units = self.config['rsu']['number_of_units']
        annual_hours = self.config['general']['annual_hours']

        E_RSU = rsu_power_kw_per_unit * rsu_number_of_units * annual_hours  # kWh
        return E_RSU

    def calculate_total_gasoline_energy_and_emissions(self):
        """
        휘발유 차량의 에너지 소비량 및 탄소 배출량 계산 (벡터화)
        Returns:
        - total_gasoline_energy: 총 휘발유 에너지 소비량 (ℓ)
        - total_carbon_gasoline: 총 휘발유 탄소 배출량 (kg CO₂)
        """
        config = self.config
        scenario = self.scenario
        average_speed = self.average_speed
        total_travel_time = self.total_travel_time

        # 운행 시간 계산
        T_gasoline = total_travel_time * scenario['vehicle_ratios'].get('gasoline', 0)

        # 휘발유차 에너지 소비량 고정
        fuel_efficiency = config['vehicles']['fuel_efficiency']['gasoline']  # 고정된 연비
        X_gasoline = (average_speed * T_gasoline) / fuel_efficiency
        total_gasoline_energy = X_gasoline
        carbon_intensity_gasoline = config['general']['carbon_intensity']['gasoline']
        total_gasoline_carbon = total_gasoline_energy * carbon_intensity_gasoline

        return total_gasoline_energy, total_gasoline_carbon
        # 휘발유차 에너지 소비량 고정
        fuel_efficiency = self.fuel_efficiency_gasoline  # 샘플링된 연비 사용
        X_gasoline = (average_speed * T_gasoline) / fuel_efficiency
        total_gasoline_energy = X_gasoline
        total_gasoline_carbon = total_gasoline_energy * self.carbon_intensity_gasoline  # 샘플링된 탄소 배출 계수 사용

        return total_gasoline_energy, total_gasoline_carbon
