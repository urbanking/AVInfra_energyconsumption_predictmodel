# models/energy_model.py

from models.vehicle_model import VehicleModel

class EnergyModel:
    def __init__(self, config, scenario, average_speed, total_travel_time,
                 fuel_efficiency_electric_samples, fuel_efficiency_gasoline_samples,
                 computing_power_samples, carbon_intensity_electric_samples,
                 carbon_intensity_gasoline_samples, efficiency_a_samples, error_term_samples):
        """
        EnergyModel 클래스 초기화
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

        # 차량 모델 인스턴스 생성
        self.vehicle_model = VehicleModel(
            config, scenario, average_speed, total_travel_time,
            fuel_efficiency_electric_samples,
            fuel_efficiency_gasoline_samples,
            computing_power_samples,
            carbon_intensity_electric_samples,
            carbon_intensity_gasoline_samples,
            efficiency_a_samples,
            error_term_samples
        )

        # RSU 에너지 소비량 계산
        self.E_RSU = self.calculate_rsu_energy()
        self.C_RSU = self.E_RSU * self.config['general']['carbon_intensity']['electric']

    def calculate_rsu_energy(self):
        """
        RSU 에너지 소비량 계산
        Returns:
        - E_RSU: RSU의 총 에너지 소비량 (kWh)
        """
        E_RSU = self.config['rsu']['power_kw_per_unit'] * \
                self.config['rsu']['number_of_units'] * \
                self.config['general']['annual_hours']
        return E_RSU

    def calculate_total_electric_energy_and_emissions(self):
        """
        전체 전기 에너지 소비량 및 탄소 배출량 계산
        Returns:
        - total_electric_energy: 총 전기 에너지 소비량 (kWh)
        - total_carbon_electric: 총 전기 탄소 배출량 (kg CO₂)
        """
        total_electric_energy, total_carbon_electric = self.vehicle_model.calculate_total_electric_energy_and_emissions()
        # RSU 전기 에너지 소비량 및 탄소 배출량 추가
        total_electric_energy += self.E_RSU
        total_carbon_electric += self.C_RSU
        return total_electric_energy, total_carbon_electric

    def calculate_total_gasoline_energy_and_emissions(self):
        """
        전체 휘발유 에너지 소비량 및 탄소 배출량 계산
        Returns:
        - total_gasoline_energy: 총 휘발유 에너지 소비량 (ℓ)
        - total_carbon_gasoline: 총 휘발유 탄소 배출량 (kg CO₂)
        """
        total_gasoline_energy, total_carbon_gasoline = self.vehicle_model.calculate_total_gasoline_energy_and_emissions()
        return total_gasoline_energy, total_carbon_gasoline
