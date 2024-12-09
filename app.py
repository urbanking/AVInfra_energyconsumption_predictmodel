# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.config_loader import load_config  # 실제 경로로 수정
from models.energy_model import EnergyModel  # 실제 경로로 수정
import numpy as np

def logistic_growth(x, K, r, x0):
    return K / (1 + np.exp(-r * (x - x0)))

def run_simulation(config, current_seoul_data):
    scenarios = config['scenarios']
    results = []
    num_simulations = 1000

    K_speed = config['general']['max_average_speed_kmh'] - config['general']['base_average_speed_kmh']
    r_speed = config['traffic_model']['logistic_k']
    x0_speed = config['traffic_model']['logistic_x0']

    K_distance = config['traffic_model']['K_distance']
    r_distance = config['traffic_model']['r_distance']
    x0_distance = config['traffic_model']['x0_distance']

    for scenario in scenarios:
        adoption_rate = scenario['adoption_rate']

        delta_speed = logistic_growth(adoption_rate, K_speed, r_speed, x0_speed)
        average_speed = config['general']['base_average_speed_kmh'] + delta_speed

        growth_factor = logistic_growth(adoption_rate, K_distance, r_distance, x0_distance)
        total_distance = config['general']['total_annual_distance_km'] * (1 + growth_factor)

        total_travel_time = total_distance / average_speed  # 시간

        ev_target_percentage = current_seoul_data['ev_adoption_targets'].get(scenario['year'], 0)
        gasoline_ratio = 1 - ev_target_percentage

        vehicle_ratios = {
            'gasoline': gasoline_ratio,
            'electric': ev_target_percentage,
            'autonomous': scenario.get('autonomous_ratio', 0)
        }

        simulation_results = {
            'total_energy_electric': [],
            'total_carbon_electric': [],
            'total_energy_gasoline': [],
            'total_carbon_gasoline': [],
            'E_gasoline': [],
            'C_gasoline': [],
            'E_electric': [],
            'C_electric': [],
            'E_drive_av': [],
            'C_drive_av': [],
            'E_compute': [],
            'E_v2x': [],
            'E_data_center': [],
            'E_RSU': [],
            'C_RSU': []
        }

        for _ in range(num_simulations):
            energy_model = EnergyModel(config, scenario, average_speed, total_travel_time, vehicle_ratios)
            total_energy_electric, total_carbon_electric = energy_model.calculate_total_electric_energy_and_emissions()
            total_energy_gasoline, total_carbon_gasoline = energy_model.calculate_total_gasoline_energy_and_emissions()

            simulation_results['total_energy_electric'].append(total_energy_electric)
            simulation_results['total_carbon_electric'].append(total_carbon_electric)
            simulation_results['total_energy_gasoline'].append(total_energy_gasoline)
            simulation_results['total_carbon_gasoline'].append(total_carbon_gasoline)
            simulation_results['E_gasoline'].append(energy_model.vehicle_model.E_gasoline)
            simulation_results['C_gasoline'].append(energy_model.vehicle_model.C_gasoline)
            simulation_results['E_electric'].append(energy_model.vehicle_model.E_electric)
            simulation_results['C_electric'].append(energy_model.vehicle_model.C_electric)
            simulation_results['E_drive_av'].append(energy_model.vehicle_model.E_drive_av)
            simulation_results['C_drive_av'].append(energy_model.vehicle_model.C_drive_av)
            simulation_results['E_compute'].append(energy_model.vehicle_model.E_compute)
            simulation_results['E_v2x'].append(energy_model.vehicle_model.E_v2x)
            simulation_results['E_data_center'].append(energy_model.vehicle_model.E_data_center)
            simulation_results['E_RSU'].append(energy_model.E_RSU)
            simulation_results['C_RSU'].append(energy_model.C_RSU)

        scenario_result = {
            'scenario': scenario['name'],
            'average_speed': average_speed,
            'total_travel_time': total_travel_time,
            'total_energy_electric_mean': np.mean(simulation_results['total_energy_electric']),
            'total_energy_electric_ci_lower': max(np.percentile(simulation_results['total_energy_electric'], 5), 0),
            'total_energy_electric_ci_upper': np.percentile(simulation_results['total_energy_electric'], 95),
            'total_carbon_electric_mean': np.mean(simulation_results['total_carbon_electric']),
            'total_carbon_electric_ci_lower': max(np.percentile(simulation_results['total_carbon_electric'], 5), 0),
            'total_carbon_electric_ci_upper': np.percentile(simulation_results['total_carbon_electric'], 95),
            'total_energy_gasoline_mean': np.mean(simulation_results['total_energy_gasoline']),
            'total_energy_gasoline_ci_lower': max(np.percentile(simulation_results['total_energy_gasoline'], 5), 0),
            'total_energy_gasoline_ci_upper': np.percentile(simulation_results['total_energy_gasoline'], 95),
            'total_carbon_gasoline_mean': np.mean(simulation_results['total_carbon_gasoline']),
            'total_carbon_gasoline_ci_lower': max(np.percentile(simulation_results['total_carbon_gasoline'], 5), 0),
            'total_carbon_gasoline_ci_upper': np.percentile(simulation_results['total_carbon_gasoline'], 95),
            'E_gasoline_mean': np.mean(simulation_results['E_gasoline']),
            'E_gasoline_ci_lower': max(np.percentile(simulation_results['E_gasoline'], 5), 0),
            'E_gasoline_ci_upper': np.percentile(simulation_results['E_gasoline'], 95),
            'C_gasoline_mean': np.mean(simulation_results['C_gasoline']),
            'C_gasoline_ci_lower': max(np.percentile(simulation_results['C_gasoline'], 5), 0),
            'C_gasoline_ci_upper': np.percentile(simulation_results['C_gasoline'], 95),
            'E_electric_mean': np.mean(simulation_results['E_electric']),
            'E_electric_ci_lower': max(np.percentile(simulation_results['E_electric'], 5), 0),
            'E_electric_ci_upper': np.percentile(simulation_results['E_electric'], 95),
            'C_electric_mean': np.mean(simulation_results['C_electric']),
            'C_electric_ci_lower': max(np.percentile(simulation_results['C_electric'], 5), 0),
            'C_electric_ci_upper': np.percentile(simulation_results['C_electric'], 95),
            'E_drive_av_mean': np.mean(simulation_results['E_drive_av']),
            'E_drive_av_ci_lower': max(np.percentile(simulation_results['E_drive_av'], 5), 0),
            'E_drive_av_ci_upper': np.percentile(simulation_results['E_drive_av'], 95),
            'C_drive_av_mean': np.mean(simulation_results['C_drive_av']),
            'C_drive_av_ci_lower': max(np.percentile(simulation_results['C_drive_av'], 5), 0),
            'C_drive_av_ci_upper': np.percentile(simulation_results['C_drive_av'], 95),
            'E_compute_mean': np.mean(simulation_results['E_compute']),
            'E_compute_ci_lower': max(np.percentile(simulation_results['E_compute'], 5), 0),
            'E_compute_ci_upper': np.percentile(simulation_results['E_compute'], 95),
            'E_v2x_mean': np.mean(simulation_results['E_v2x']),
            'E_v2x_ci_lower': max(np.percentile(simulation_results['E_v2x'], 5), 0),
            'E_v2x_ci_upper': np.percentile(simulation_results['E_v2x'], 95),
            'E_data_center_mean': np.mean(simulation_results['E_data_center']),
            'E_data_center_ci_lower': max(np.percentile(simulation_results['E_data_center'], 5), 0),
            'E_data_center_ci_upper': np.percentile(simulation_results['E_data_center'], 95),
            'E_RSU_mean': np.mean(simulation_results['E_RSU']),
            'E_RSU_ci_lower': max(np.percentile(simulation_results['E_RSU'], 5), 0),
            'E_RSU_ci_upper': np.percentile(simulation_results['E_RSU'], 95),
            'C_RSU_mean': np.mean(simulation_results['C_RSU']),
            'C_RSU_ci_lower': max(np.percentile(simulation_results['C_RSU'], 5), 0),
            'C_RSU_ci_upper': np.percentile(simulation_results['C_RSU'], 95)
        }

        results.append(scenario_result)

    return results

def plot_results_streamlit(results, current_seoul_data):
    """
    Streamlit을 사용하여 시뮬레이션 결과 시각화
    """
    scenarios = [res['scenario'] for res in results]
    
    # 데이터 프레임 생성
    df = pd.DataFrame(results)

    # 총 전기 에너지 소비량 및 구성 요소 스택형 바차트
    st.header("총 전기 에너지 소비량 및 구성 요소")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(scenarios, df['E_electric_mean'], label='Electric Vehicles (kWh)', color='skyblue')
    ax.bar(scenarios, df['E_drive_av_mean'], bottom=df['E_electric_mean'], label='AV Driving Energy (kWh)', color='steelblue')
    bottom_E = df['E_electric_mean'] + df['E_drive_av_mean']
    ax.bar(scenarios, df['E_compute_mean'], bottom=bottom_E, label='AV Compute Energy (kWh)', color='dodgerblue')
    bottom_E += df['E_compute_mean']
    ax.bar(scenarios, df['E_v2x_mean'], bottom=bottom_E, label='AV V2X Energy (kWh)', color='deepskyblue')
    bottom_E += df['E_v2x_mean']
    ax.bar(scenarios, df['E_data_center_mean'], bottom=bottom_E, label='AV Data Center Energy (kWh)', color='lightskyblue')
    bottom_E += df['E_data_center_mean']
    ax.bar(scenarios, df['E_RSU_mean'], bottom=bottom_E, label='RSU Energy (kWh)', color='royalblue')
    
    ax.set_xlabel('시나리오')
    ax.set_ylabel('에너지 소비량 (kWh)')
    ax.set_title('총 전기 에너지 소비량 및 구성 요소')
    ax.legend()
    st.pyplot(fig)

    # 탄소 배출량 시각화
    st.header("총 탄소 배출량")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(scenarios, df['C_electric_mean'], label='Electric Vehicles (kg CO₂)', color='lightgreen')
    ax2.bar(scenarios, df['C_gasoline_mean'], bottom=df['C_electric_mean'], label='Gasoline Vehicles (kg CO₂)', color='salmon')
    
    # 신뢰구간 표시
    ax2.errorbar(scenarios, df['C_electric_mean'], 
                yerr=[df['C_electric_mean'] - df['C_electric_ci_lower'], 
                      df['C_electric_ci_upper'] - df['C_electric_mean']], 
                fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    ax2.errorbar(scenarios, df['C_electric_mean'] + df['C_gasoline_mean'], 
                yerr=[df['C_gasoline_mean'] - df['C_gasoline_ci_lower'], 
                      df['C_gasoline_ci_upper'] - df['C_gasoline_mean']], 
                fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    
    # 현재 서울의 탄소 배출량 기준선 추가
    if 'total_carbon_electric' in current_seoul_data:
        ax2.axhline(y=current_seoul_data['total_carbon_electric'], color='blue', linestyle='--', label='현재 서울 전기차 탄소 배출량')
    if 'total_carbon_gasoline' in current_seoul_data:
        ax2.axhline(y=current_seoul_data['total_carbon_gasoline'], color='orange', linestyle='--', label='현재 서울 휘발유차 탄소 배출량')
    
    ax2.set_xlabel('시나리오')
    ax2.set_ylabel('탄소 배출량 (kg CO₂)')
    ax2.set_title('총 탄소 배출량')
    ax2.legend()
    st.pyplot(fig2)

    # 총 이동 시간 시각화
    st.header("총 이동 시간")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(scenarios, df['total_travel_time'], color='mediumpurple', label='Total Travel Time (hours)')
    
    if 'total_travel_time' in current_seoul_data:
        ax3.axhline(y=current_seoul_data['total_travel_time'], color='red', linestyle='--', label='현재 서울 총 이동 시간')
    
    ax3.set_xlabel('시나리오')
    ax3.set_ylabel('총 이동 시간 (시간)')
    ax3.set_title('총 이동 시간')
    ax3.legend()
    st.pyplot(fig3)

    # 평균 속도 시각화
    st.header("평균 속도")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.bar(scenarios, df['average_speed'], color='teal', label='Average Speed (km/h)')
    
    if 'average_speed' in current_seoul_data:
        ax4.axhline(y=current_seoul_data['average_speed'], color='red', linestyle='--', label='현재 서울 평균 속도')
    
    ax4.set_xlabel('시나리오')
    ax4.set_ylabel('평균 속도 (km/h)')
    ax4.set_title('평균 속도')
    ax4.legend()
    st.pyplot(fig4)

def main_app():
    st.title("서울시 미래 모빌리티 시뮬레이션 대시보드")
    
    # 디버깅 정보 출력
    st.write("현재 작업 디렉토리:", os.getcwd())
    st.write("디렉토리 내 파일들:", os.listdir('.'))
    
    # 설정 파일 로드
    try:
        config = load_config('config.yaml')  # 올바른 경로로 수정
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"설정 파일을 로드하는 중 오류가 발생했습니다: {e}")
        st.stop()
    
    # 현재 서울 데이터 설정 (업데이트된 데이터 사용)
    current_seoul_data = {
        'baseline_year': 2005,  # 기준 연도
        'current_year': 2023,  # 현재 연도
        'total_energy_electric': 26000000000,  # kWh (기존 데이터 유지)
        'total_carbon_electric': 11000000000,  # kg CO₂ (기존 데이터 유지)
        'total_energy_gasoline': 0,  # ℓ (휘발유는 전력과 별개로 처리)
        'total_carbon_gasoline': 0,  # kg CO₂ (휘발유는 전력과 별개로 처리)
        'current_energy_compute': 50000000,  # kWh (예시)
        'current_energy_v2x': 20000000,      # kWh (예시)
        'current_energy_data_center': 100000000,  # kWh (예시)
        'average_speed': 23,  # 예시 값 추가
        'total_travel_time': 38603400000 / 23,  # 예시 값 계산
        'power_supply_targets': {
            2025: 9899,   # GWh
            2030: 12061,  # GWh
            2040: 16221   # GWh
        },
        'power_generation_targets_2040': {
            'solar': 4088,            # GWh
            'fuel_cell_generation': 3548,  # GWh
            'fuel_cell_building': 2759,    # GWh
            'seoul_coal_power': 4205,      # GWh
            'magok_CHP': 998.6             # GWh
        },
        'ghg_emission_targets': {
            2025: 38654,  # 천톤 CO₂eq
            2030: 29349,  # 천톤 CO₂eq
            2040: 14375   # 천톤 CO₂eq
        },
        'ev_adoption_targets': {
            2025: 100000,  # 대
            2030: 0.23,     # 비율 (23%)
            2040: 0.35,     # 비율 (35%)
            2050: 0.982     # 비율 (98.2%)
        },
        'ev_conversion_targets_2050': {
            'taxi': 1.0,   # 100%
            'passenger_car': 0.98,  # 98%
            'bus': 0.97    # 97%
        },
        'energy_reduction_targets': {
            2040: 1544  # 천 TOE
        },
        'transport_energy_reduction_percent': 0.64,  # 64% 감축 (2005년 대비)
        'charging_infrastructure_targets': {
            2025: 2700  # 기
        },
        'policy_support': {
            'purchase_subsidy': True,
            'additional_subsidy_for_conversion': True
        }
    }

    # 시뮬레이션 실행 버튼
    if st.button('시뮬레이션 실행'):
        with st.spinner('시뮬레이션 중...'):
            try:
                results = run_simulation(config, current_seoul_data)
                df = pd.DataFrame(results)
                df.to_csv('simulation_results.csv', index=False)
                st.success('시뮬레이션 완료!')
                plot_results_streamlit(results, current_seoul_data)
            except Exception as e:
                st.error(f"시뮬레이션 실행 중 오류가 발생했습니다: {e}")
    
    # 시뮬레이션 결과가 존재하는 경우 시각화 표시
    data_filepath = 'simulation_results.csv'
    if os.path.exists(data_filepath):
        data = pd.read_csv(data_filepath)
        if not data.empty:
            st.header("기존 시뮬레이션 결과 시각화")
            scenarios = data['scenario'].unique().tolist()
            selected_scenario = st.selectbox("시나리오 선택 (기존)", scenarios)

            # 선택된 시나리오 데이터 필터링
            scenario_data = data[data['scenario'] == selected_scenario].iloc[0]

            # 탄소 배출량 그래프
            st.subheader("탄소 배출량")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.bar(['Electric Vehicles', 'Gasoline Vehicles'], 
                    [scenario_data['total_carbon_electric_mean'], scenario_data['total_carbon_gasoline_mean']], 
                    color=['lightgreen', 'salmon'])
            ax1.set_ylabel('탄소 배출량 (kg CO₂)')
            ax1.set_title('전기차 vs 휘발유차 탄소 배출량')
            st.pyplot(fig1)

            # 에너지 소비량 그래프
            st.subheader("에너지 소비량")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.bar(['Electric Energy', 'Gasoline Energy'], 
                    [scenario_data['total_energy_electric_mean'], scenario_data['total_energy_gasoline_mean']], 
                    color=['skyblue', 'orange'])
            ax2.set_ylabel('에너지 소비량 (kWh / ℓ)')
            ax2.set_title('전기 에너지 vs 휘발유 에너지 소비량')
            st.pyplot(fig2)

            # 총 이동 시간 그래프
            st.subheader("총 이동 시간")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.bar(['Total Travel Time'], [scenario_data['total_travel_time']], color=['mediumpurple'])
            ax3.set_ylabel('총 이동 시간 (시간)')
            ax3.set_title('총 이동 시간')
            st.pyplot(fig3)

            # 평균 속도 그래프
            st.subheader("평균 속도")
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            ax4.bar(['Average Speed'], [scenario_data['average_speed']], color=['teal'])
            ax4.set_ylabel('평균 속도 (km/h)')
            ax4.set_title('평균 속도')
            st.pyplot(fig4)

            # 기준선 비교 그래프
            st.subheader("현재 서울의 기준선과 비교")
            
            # 탄소 배출량 비교 그래프
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            ax5.bar(['Electric Vehicles', 'Gasoline Vehicles'], 
                    [scenario_data['total_carbon_electric_mean'], scenario_data['total_carbon_gasoline_mean']], 
                    color=['lightgreen', 'salmon'], label='시나리오')
            ax5.axhline(y=current_seoul_data['total_carbon_electric'], color='blue', linestyle='--', label='현재 서울 전기차 탄소 배출량')
            ax5.axhline(y=current_seoul_data['total_carbon_gasoline'], color='orange', linestyle='--', label='현재 서울 휘발유차 탄소 배출량')
            ax5.set_ylabel('탄소 배출량 (kg CO₂)')
            ax5.set_title('탄소 배출량 비교')
            ax5.legend()
            st.pyplot(fig5)

            # 에너지 소비량 비교 그래프
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            ax6.bar(['Electric Energy', 'Gasoline Energy'], 
                    [scenario_data['total_energy_electric_mean'], scenario_data['total_energy_gasoline_mean']], 
                    color=['skyblue', 'orange'], label='시나리오')
            ax6.axhline(y=current_seoul_data['total_energy_electric'], color='blue', linestyle='--', label='현재 서울 전기 에너지 소비량')
            ax6.axhline(y=current_seoul_data['total_energy_gasoline'], color='orange', linestyle='--', label='현재 서울 휘발유 에너지 소비량')
            ax6.set_ylabel('에너지 소비량 (kWh / ℓ)')
            ax6.set_title('에너지 소비량 비교')
            ax6.legend()
            st.pyplot(fig6)

            # 총 이동 시간 비교 그래프
            fig7, ax7 = plt.subplots(figsize=(10, 6))
            ax7.bar(['Total Travel Time'], [scenario_data['total_travel_time']], color=['mediumpurple'], label='시나리오')
            ax7.axhline(y=current_seoul_data['total_travel_time'], color='red', linestyle='--', label='현재 서울 총 이동 시간')
            ax7.set_ylabel('총 이동 시간 (시간)')
            ax7.set_title('총 이동 시간 비교')
            ax7.legend()
            st.pyplot(fig7)

            # 평균 속도 비교 그래프
            fig8, ax8 = plt.subplots(figsize=(10, 6))
            ax8.bar(['Average Speed'], [scenario_data['average_speed']], color=['teal'], label='시나리오')
            ax8.axhline(y=current_seoul_data['average_speed'], color='red', linestyle='--', label='현재 서울 평균 속도')
            ax8.set_ylabel('평균 속도 (km/h)')
            ax8.set_title('평균 속도 비교')
            ax8.legend()
            st.pyplot(fig8)

    else:
        st.error(f"시뮬레이션 결과 파일을 찾을 수 없습니다: {data_filepath}")
        st.info("먼저 `main.py`를 실행하여 `simulation_results.csv` 파일을 생성하세요.")

if __name__ == '__main__':
    main_app()
