import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 추가
import seaborn as sns            # 추가

from utils.config_loader import load_config
from models.vehicle_model import VehicleModel
from utils.visualization import plot_results, plot_energy_consumption_breakdown
from itertools import product

def logistic_growth(x, K, r, x0):
    """
    Logistic growth function.
    Parameters:
    - x: Adoption rate (0 ~ 1)
    - K: Carrying capacity (growth limit)
    - r: Growth rate
    - x0: Midpoint of the growth curve
    Returns:
    - Logistic growth value
    """
    return K / (1 + np.exp(-r * (x - x0)))

def generate_scenarios(base_config):
    """
    energy_reduction의 다양한 레벨을 고려하여 시나리오 생성.
    """
    scenarios = []
    adoption_rates = base_config.get('scenarios', {}).get('adoption_rates', range(0, 101, 10))
    efficiency_a_values = base_config.get('scenarios', {}).get('efficiency_a_values', [1.0])
    clustering_levels = base_config['general']['energy_reduction']['levels']['clustering']
    optimal_routing_levels = base_config['general']['energy_reduction']['levels']['optimal_routing']
    num_combinations_per_rate = base_config.get('scenarios', {}).get('num_combinations_per_rate', 10)
    
    # 파라미터 조합 생성
    for a_value, rate, clustering, optimal_routing in product(
        efficiency_a_values, adoption_rates, clustering_levels, optimal_routing_levels
    ):
        adoption_rate = rate / 100.0
        
        for _ in range(num_combinations_per_rate):
            # 고정된 차량 비율 반영
            gasoline_ratio = 0.95 * (1 - adoption_rate)
            electric_ratio = 0.05 * (1 - adoption_rate)
            autonomous_ratio = adoption_rate
            
            scenario = {
                'name': f"a {a_value} - adoption {rate}% - Gasoline {gasoline_ratio:.2f}, Electric {electric_ratio:.2f}, Autonomous {autonomous_ratio:.2f} - Clustering {clustering}, Routing {optimal_routing}",
                'adoption_rate': adoption_rate,
                'efficiency_a': a_value,
                'energy_reduction': {
                    'clustering': clustering,
                    'optimal_routing': optimal_routing
                },
                'vehicle_ratios': {
                    'gasoline': gasoline_ratio,
                    'electric': electric_ratio,
                    'autonomous': autonomous_ratio
                }
            }
            scenarios.append(scenario)
    return scenarios

def run_simulation(config, scenario, num_simulations=1000):
    """
    몬테카를로 방식으로 시뮬레이션을 실행.
    """
    adoption_rate = scenario['adoption_rate']
    efficiency_a = scenario.get('efficiency_a', 1.0)

    # 로지스틱 성장 계산
    delta_speed = logistic_growth(adoption_rate,
                                   config['general']['max_average_speed_kmh'] - config['general']['base_average_speed_kmh'],
                                   config['traffic_model']['logistic_k'],
                                   config['traffic_model']['logistic_x0'])
    average_speed = config['general']['base_average_speed_kmh'] + delta_speed

    growth_factor = logistic_growth(adoption_rate,
                                     config['traffic_model']['K_distance'],
                                     config['traffic_model']['r_distance'],
                                     config['traffic_model']['x0_distance'])
    total_distance = config['general']['total_annual_distance_km'] * (1 + growth_factor)

    total_travel_time = total_distance / average_speed

    # 벡터화된 시뮬레이션
    simulation_count = num_simulations

    # 샘플 파라미터 생성
    electric_efficiency_samples = np.random.normal(
        config['general']['uncertainty']['fuel_efficiency_electric']['mean'],
        config['general']['uncertainty']['fuel_efficiency_electric']['std_dev'],
        simulation_count
    )
    computing_power_samples = np.random.normal(
        config['general']['uncertainty']['computing_power']['mean'],
        config['general']['uncertainty']['computing_power']['std_dev'],
        simulation_count
    )
    efficiency_a_samples = np.full(num_simulations, efficiency_a)
    error_term_samples = np.random.normal(
        config['general']['uncertainty']['error_term']['mean'],
        config['general']['uncertainty']['error_term']['std_dev'],
        simulation_count
    )

    # 모든 불확실한 파라미터 샘플링
    uncertainty = config['general']['uncertainty']
    
    fuel_efficiency_gasoline_samples = np.random.normal(
        uncertainty['fuel_efficiency_gasoline']['mean'],
        uncertainty['fuel_efficiency_gasoline']['std_dev'],
        num_simulations
    )
    fuel_efficiency_gasoline_samples = np.maximum(fuel_efficiency_gasoline_samples, uncertainty['fuel_efficiency_gasoline']['min'])
    
    fuel_efficiency_electric_samples = np.random.normal(
        uncertainty['fuel_efficiency_electric']['mean'],
        uncertainty['fuel_efficiency_electric']['std_dev'],
        num_simulations
    )
    fuel_efficiency_electric_samples = np.maximum(fuel_efficiency_electric_samples, uncertainty['fuel_efficiency_electric']['min'])
    
    computing_power_samples = np.random.normal(
        uncertainty['computing_power']['mean'],
        uncertainty['computing_power']['std_dev'],
        num_simulations
    )
    computing_power_samples = np.maximum(computing_power_samples, uncertainty['computing_power']['min'])
    
    carbon_intensity_gasoline_samples = np.random.normal(
        uncertainty['carbon_intensity_gasoline']['mean'],
        uncertainty['carbon_intensity_gasoline']['std_dev'],
        num_simulations
    )
    carbon_intensity_gasoline_samples = np.maximum(carbon_intensity_gasoline_samples, uncertainty['carbon_intensity_gasoline']['min'])
    
    carbon_intensity_electric_samples = np.random.normal(
        uncertainty['carbon_intensity_electric']['mean'],
        uncertainty['carbon_intensity_electric']['std_dev'],
        num_simulations
    )
    carbon_intensity_electric_samples = np.maximum(carbon_intensity_electric_samples, uncertainty['carbon_intensity_electric']['min'])
    
    efficiency_a_samples = np.random.normal(
        uncertainty['efficiency_a']['mean'],
        uncertainty['efficiency_a']['std_dev'],
        num_simulations
    )
    efficiency_a_samples = np.maximum(efficiency_a_samples, uncertainty['efficiency_a']['min'])
    
    error_term_samples = np.random.normal(
        uncertainty['error_term']['mean'],
        uncertainty['error_term']['std_dev'],
        num_simulations
    )
    error_term_samples = np.maximum(error_term_samples, uncertainty['error_term']['min'])

    # energy_reduction 레벨 적용
    clustering_reduction = scenario['energy_reduction']['clustering']
    routing_reduction = scenario['energy_reduction']['optimal_routing']
    energy_reduction = clustering_reduction + routing_reduction
    energy_reduction = min(energy_reduction, 
                           config['general']['energy_reduction']['clustering'] +
                           config['general']['energy_reduction']['optimal_routing'])

    # VehicleModel 인스턴스 생성 (벡터화된 입력 사용)
    vehicle_model = VehicleModel(
        config, scenario, average_speed, total_travel_time,
        fuel_efficiency_electric_samples,
        fuel_efficiency_gasoline_samples,
        computing_power_samples,
        carbon_intensity_electric_samples,
        carbon_intensity_gasoline_samples,
        efficiency_a_samples,
        error_term_samples
    )

    # 에너지 및 배출량 계산
    total_electric_energy, total_electric_carbon = vehicle_model.calculate_total_electric_energy_and_emissions()
    total_gasoline_energy, total_gasoline_carbon = vehicle_model.calculate_total_gasoline_energy_and_emissions()

    # 각 에너지 구성 요소의 평균 계산
    E_electric_mean = np.mean(vehicle_model.E_electric)
    E_drive_av_mean = np.mean(vehicle_model.E_drive_av)
    E_compute_mean = np.mean(vehicle_model.E_compute)
    E_v2x_mean = np.mean(vehicle_model.E_v2x)
    E_data_center_mean = np.mean(vehicle_model.E_data_center)
    # 필요한 경우 E_RSU_mean도 추가
    # E_RSU_mean = np.mean(vehicle_model.E_RSU)

    # E_RSU_mean 계산
    E_RSU_mean = vehicle_model.E_RSU  # 단일 값이므로 평균 계산 필요 없음

    # 결과 수집
    result = {
        'scenario': scenario['name'],
        'adoption_rate': scenario['adoption_rate'],
        'efficiency_a': efficiency_a,
        'vehicle_ratios': scenario['vehicle_ratios'],
        'average_speed': average_speed,
        'total_travel_time': total_travel_time,
        'total_energy_electric_mean': np.mean(total_electric_energy),
        'total_carbon_electric_mean': np.mean(total_electric_carbon),
        'total_energy_gasoline_mean': np.mean(total_gasoline_energy),
        'total_carbon_gasoline_mean': np.mean(total_gasoline_carbon),
        'E_electric_mean': E_electric_mean,
        'E_drive_av_mean': E_drive_av_mean,
        'E_compute_mean': E_compute_mean,
        'E_v2x_mean': E_v2x_mean,
        'E_data_center_mean': E_data_center_mean,
        'E_RSU_mean': E_RSU_mean,
    }

    # 신뢰 구간 계산 및 결과에 추가
    for key, data_array in [
        ('total_energy_electric', total_electric_energy),
        ('total_carbon_electric', total_electric_carbon),
        ('total_energy_gasoline', total_gasoline_energy),
        ('total_carbon_gasoline', total_gasoline_carbon),
        ('E_electric', vehicle_model.E_electric),
        ('E_drive_av', vehicle_model.E_drive_av),
        ('E_compute', vehicle_model.E_compute),
        ('E_v2x', vehicle_model.E_v2x),
        ('E_data_center', vehicle_model.E_data_center),
        # 필요한 경우 E_RSU도 추가
        # ('E_RSU', vehicle_model.E_RSU),
    ]:
        result[f"{key}_ci_lower"] = np.percentile(data_array, 5)
        result[f"{key}_ci_upper"] = np.percentile(data_array, 95)
        # 이미 평균을 계산한 경우 중복 계산 방지
        if f"{key}_mean" not in result:
            result[f"{key}_mean"] = np.mean(data_array)

    return result

def perform_sensitivity_analysis(df, parameter):
    """
    민감도 분석을 수행하는 함수.
    Parameters:
    - df: 시뮬레이션 결과 DataFrame
    - parameter: 민감도 분석할 파라미터 (str)
    Returns:
    - 민감도 분석 결과
    """
    numeric_df = df.select_dtypes(include=[np.number])  # 숫자형 컬럼만 선택
    correlation = numeric_df.corr()[parameter]
    return correlation

def main():
    # Load configuration
    config = load_config('config.yaml')

    # Generate scenarios with energy_reduction levels
    scenarios = generate_scenarios(config)
    print(f"Total scenarios generated: {len(scenarios)}")

    # Run simulations for all scenarios using Monte Carlo method
    results = []
    for scenario in scenarios:
        print(f"Running simulation for {scenario['name']} with vehicle ratios: {scenario['vehicle_ratios']} and efficiency_a: {scenario['efficiency_a']}...")
        result = run_simulation(config, scenario)
        results.append(result)
        
        # === 시나리오 결과 출력 ===
        print(f"=== {scenario['name']} ===")
        print(f"평균 속도: {result['average_speed']:.2f} km/h")
        print(f"총 운행 시간: {result['total_travel_time']:,} 시간")
        print(f"총 에너지 소비량: {result['total_energy_electric_mean']:,} kWh")
        print(f"탄소 배출량: {result['total_carbon_electric_mean']:,} kg CO₂\n")

    # 결과를 DataFrame으로 저장하기 전에 시나리오 가정을 펼쳐서 추가
    df_results = pd.DataFrame(results)
    df_vehicle_ratios = df_results['vehicle_ratios'].apply(pd.Series)
    df_results = pd.concat([df_results.drop('vehicle_ratios', axis=1), df_vehicle_ratios], axis=1)

    # 민감도 분석 수행
    sensitivity_results = perform_sensitivity_analysis(df_results, 'total_energy_electric_mean')
    print("민감도 분석 결과:")
    print(sensitivity_results)

    # 결과를 CSV로 저장
    output_file = 'simulation_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Visualize results
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_results(results, config, current_seoul_data={})
    # plot_energy_consumption_breakdown(results, config, output_dir, [res['scenario'] for res in results])  # 제거됨

if __name__ == '__main__':
    main()
