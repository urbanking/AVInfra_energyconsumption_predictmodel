# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_results(results, config, current_seoul_data=None):
    """
    시나리오별 에너지 소비량과 탄소 배출량, 총 이동 시간, 평균 속도를 시각화
    효율성 계수 a에 따라 그래프를 병렬로 작성합니다.
    
    Parameters:
    - results: 시나리오 결과가 담긴 딕셔너리의 리스트.
    - current_seoul_data: 현재 서울의 예상치를 담은 딕셔너리 (선택 사항).
    """
    # 효율성 계수 a별로 결과를 그룹핑
    a_values = sorted(set(res['efficiency_a'] for res in results))
    scenarios_by_a = {a: [] for a in a_values}
    for res in results:
        a = res['efficiency_a']
        scenarios_by_a[a].append(res)
    
    for a, a_results in scenarios_by_a.items():
        if not a_results:
            continue  # a_results가 비어 있으면 다음으로 넘어갑니다.

        scenarios = [res['scenario'] for res in a_results]
        
        # 에너지 절감 효과 고정 비율 적용
        energy_reduction = 0.2 + 0.07  # 군집 주행 20%, 최적 경로 7%
        energy_reduction = min(energy_reduction, 
                               config['general']['energy_reduction']['clustering'] +
                               config['general']['energy_reduction']['optimal_routing'])
        
        # 에너지 소비 데이터
        total_energy_electric_mean = [res['total_energy_electric_mean'] for res in a_results]
        total_energy_electric_ci_lower = [res['total_energy_electric_ci_lower'] for res in a_results]
        total_energy_electric_ci_upper = [res['total_energy_electric_ci_upper'] for res in a_results]
        
        total_energy_gasoline_mean = [res['total_energy_gasoline_mean'] for res in a_results]
        total_energy_gasoline_ci_lower = [res['total_energy_gasoline_ci_lower'] for res in a_results]
        total_energy_gasoline_ci_upper = [res['total_energy_gasoline_ci_upper'] for res in a_results]
        
        # 탄소 배출 데이터
        total_carbon_electric_mean = [res['total_carbon_electric_mean'] for res in a_results]
        total_carbon_electric_ci_lower = [res['total_carbon_electric_ci_lower'] for res in a_results]
        total_carbon_electric_ci_upper = [res['total_carbon_electric_ci_upper'] for res in a_results]
        
        total_carbon_gasoline_mean = [res['total_carbon_gasoline_mean'] for res in a_results]
        total_carbon_gasoline_ci_lower = [res['total_carbon_gasoline_ci_lower'] for res in a_results]
        total_carbon_gasoline_ci_upper = [res['total_carbon_gasoline_ci_upper'] for res in a_results]
        
        # 총 이동 시간 및 평균 속도 데이터
        total_travel_time = [res['total_travel_time'] for res in a_results]
        average_speed = [res['average_speed'] for res in a_results]
        
        # 신뢰구간 범위 계산 (음수 값 제거)
        total_energy_electric_error = [
            [max(mean - lower, 0), max(upper - mean, 0)] 
            for mean, lower, upper in zip(total_energy_electric_mean, total_energy_electric_ci_lower, total_energy_electric_ci_upper)
        ]
        
        total_carbon_electric_error = [
            [max(mean - lower, 0), max(upper - mean, 0)]
            for mean, lower, upper in zip(total_carbon_electric_mean, total_carbon_electric_ci_lower, total_carbon_electric_ci_upper)
        ]
        
        total_carbon_gasoline_error = [
            [max(mean - lower, 0), max(upper - mean, 0)] 
            for mean, lower, upper in zip(total_carbon_gasoline_mean, total_carbon_gasoline_ci_lower, total_carbon_gasoline_ci_upper)
        ]
        
        # 1. 총 전기 에너지 소비량 및 구성 요소 스택형 바차트
        plt.figure(figsize=(20, 10))
        ind = np.arange(len(scenarios))
        width = 0.6

        E_electric_mean = [res.get('E_electric_mean', 0) for res in a_results]
        E_drive_av_mean = [res.get('E_drive_av_mean', 0) for res in a_results]
        E_compute_mean = [res.get('E_compute_mean', 0) for res in a_results]
        E_v2x_mean = [res.get('E_v2x_mean', 0) for res in a_results]
        E_data_center_mean = [res.get('E_data_center_mean', 0) for res in a_results]
        E_RSU_mean = [res['E_RSU_mean'] for res in a_results]

        plt.bar(ind, E_electric_mean, width, label='Electric Vehicles (kWh)', color='green')  # 기존 'skyblue'에서 'green'으로 변경
        plt.bar(ind, E_drive_av_mean, width, bottom=E_electric_mean, label='AV Driving Energy (kWh)', color='blue')  # 기존 'steelblue'에서 'blue'으로 변경
        bottom_E = np.array(E_electric_mean) + np.array(E_drive_av_mean)
        plt.bar(ind, E_compute_mean, width, bottom=bottom_E, label='AV Compute Energy (kWh)', color='orange')  # 기존 'dodgerblue'에서 'orange'으로 변경
        bottom_E += np.array(E_compute_mean)
        plt.bar(ind, E_v2x_mean, width, bottom=bottom_E, label='AV V2X Energy (kWh)', color='purple')  # 기존 'deepskyblue'에서 'purple'으로 변경
        bottom_E += np.array(E_v2x_mean)
        plt.bar(ind, E_data_center_mean, width, bottom=bottom_E, label='AV Data Center Energy (kWh)', color='brown')  # 기존 'lightskyblue'에서 'brown'으로 변경
        bottom_E += np.array(E_data_center_mean)
        plt.bar(ind, E_RSU_mean, width, bottom=bottom_E, label='RSU Energy (kWh)', color='grey')  # 기존 'royalblue'에서 'grey'으로 변경

        # 신뢰구간 표시
        plt.errorbar(ind, total_energy_electric_mean, 
                     yerr=np.array(total_energy_electric_error).T,
                     fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, label='90% Confidence Interval')

        if current_seoul_data and 'total_energy_electric' in current_seoul_data:
            plt.axhline(y=current_seoul_data['total_energy_electric'], color='red', linestyle='--', label='Current Seoul Electric Estimate')

        plt.ylabel('Energy Consumption (kWh)')
        plt.xlabel('Scenario')
        plt.title(f'Total Electric Energy Consumption and Its Components by Scenario with 90% Confidence Interval (Efficiency A = {a})')
        
        # X축 레이블을 10개만 표시 (0%, 10%, ..., 100%)
        tick_indices = np.linspace(0, len(scenarios)-1, 11, dtype=int)
        tick_labels = [scenarios[i] for i in tick_indices]
        plt.xticks(tick_indices, tick_labels, rotation=45)
        
        plt.legend()
        plt.tight_layout()
        output_dir = 'figure'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'total_electric_energy_consumption_a_{a}.png'))
        plt.show()

        # 2. ��기 에너지 소비 vs 휘발유 에너지 소비 비교 단위를 탄소 배출량으로 통일
        fig, ax = plt.subplots(figsize=(20, 10))
        bar_width = 0.35
        ind = np.arange(len(scenarios))

        # 전기 탄소 배출량의 평균과 신뢰구간
        carbon_electric_mean = [res['total_carbon_electric_mean'] for res in a_results]
        carbon_electric_ci_lower = [res['total_carbon_electric_ci_lower'] for res in a_results]
        carbon_electric_ci_upper = [res['total_carbon_electric_ci_upper'] for res in a_results]
        carbon_electric_errors = np.array([
            [res['total_carbon_electric_mean'] - res['total_carbon_electric_ci_lower'],
             res['total_carbon_electric_ci_upper'] - res['total_carbon_electric_mean']]
            for res in a_results
        ]).T

        ax.bar(ind - bar_width/2, carbon_electric_mean, bar_width, 
               yerr=carbon_electric_errors,
               capsize=5, label='Total Electric Carbon Emissions (kg CO₂)', color='green', alpha=0.7, ecolor='black')

        # 휘발유 탄소 배출량의 평균과 신뢰구간
        carbon_gasoline_mean = [res['total_carbon_gasoline_mean'] for res in a_results]
        carbon_gasoline_ci_lower = [res['total_carbon_gasoline_ci_lower'] for res in a_results]
        carbon_gasoline_ci_upper = [res['total_carbon_gasoline_ci_upper'] for res in a_results]
        carbon_gasoline_errors = np.array([
            [res['total_carbon_gasoline_mean'] - res['total_carbon_gasoline_ci_lower'],
             res['total_carbon_gasoline_ci_upper'] - res['total_carbon_gasoline_mean']]
            for res in a_results
        ]).T

        ax.bar(ind + bar_width/2, carbon_gasoline_mean, bar_width, 
               yerr=carbon_gasoline_errors,
               capsize=5, label='Total Gasoline Carbon Emissions (kg CO₂)', color='red', alpha=0.7, ecolor='black')

        if current_seoul_data:
            if 'total_carbon_electric' in current_seoul_data:
                ax.axhline(y=current_seoul_data['total_carbon_electric'], color='blue', linestyle='--', label='Current Seoul Electric Carbon Estimate')
            if 'total_carbon_gasoline' in current_seoul_data:
                ax.axhline(y=current_seoul_data['total_carbon_gasoline'], color='red', linestyle='--', label='Current Seoul Gasoline Carbon Estimate')

        ax.set_xlabel('Scenario')
        ax.set_ylabel('Carbon Emissions (kg CO₂)')
        ax.set_title(f'Comparison of Total Electric and Gasoline Carbon Emissions by Scenario with 90% Confidence Interval (Efficiency A = {a})')

        # X축 레이블을 10개만 표시
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45)

        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'electric_vs_gasoline_carbon_comparison_a_{a}.png'))
        plt.show()

        # 3. 탄소 배출량 시각화
        plt.figure(figsize=(20, 10))
        ind = np.arange(len(scenarios))
        width = 0.6

        C_electric_mean = [res['total_carbon_electric_mean'] for res in a_results]
        C_electric_ci_lower = [res['total_carbon_electric_ci_lower'] for res in a_results]
        C_electric_ci_upper = [res['total_carbon_electric_ci_upper'] for res in a_results]
        
        C_gasoline_mean = [res['total_carbon_gasoline_mean'] for res in a_results]
        C_gasoline_ci_lower = [res['total_carbon_gasoline_ci_lower'] for res in a_results]
        C_gasoline_ci_upper = [res['total_carbon_gasoline_ci_upper'] for res in a_results]

        C_electric_error = [
            [max(mean - lower, 0), max(upper - mean, 0)] 
            for mean, lower, upper in zip(C_electric_mean, C_electric_ci_lower, C_electric_ci_upper)
        ]

        C_gasoline_error = [
            [max(mean - lower, 0), max(upper - mean, 0)] 
            for mean, lower, upper in zip(C_gasoline_mean, C_gasoline_ci_lower, C_gasoline_ci_upper)
        ]

        plt.bar(ind, C_electric_mean, width, label='Electric Vehicles (kg CO₂)', color='darkgreen')  # 기존 'lightgreen'에서 'darkgreen'으로 변경
        plt.bar(ind, C_gasoline_mean, width, bottom=C_electric_mean, label='Gasoline Vehicles (kg CO₂)', color='maroon')  # 기존 'salmon'에서 'maroon'으로 변경

        # 신뢰구간 표시
        plt.errorbar(ind, C_electric_mean, 
                     yerr=np.array(C_electric_error).T,
                     fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, label='90% CI Electric')
        plt.errorbar(ind, np.array(C_electric_mean) + np.array(C_gasoline_mean), 
                     yerr=np.array(C_gasoline_error).T,
                     fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, label='90% CI Gasoline')

        if current_seoul_data and 'total_carbon_electric' in current_seoul_data:
            plt.axhline(y=current_seoul_data['total_carbon_electric'], color='blue', linestyle='--', label='Current Seoul Electric Carbon Estimate')
        if current_seoul_data and 'total_carbon_gasoline' in current_seoul_data:
            plt.axhline(y=current_seoul_data['total_carbon_gasoline'], color='red', linestyle='--', label='Current Seoul Gasoline Carbon Estimate')

        plt.ylabel('Carbon Emissions (kg CO₂)')
        plt.xlabel('Scenario')
        plt.title(f'Total Carbon Emissions by Scenario with 90% Confidence Interval (Efficiency A = {a})')

        # X축 레이블을 10개만 표시
        plt.xticks(tick_indices, tick_labels, rotation=45)

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'total_carbon_emissions_a_{a}.png'))
        plt.show()

        # 4. 총 이동 시간 시각화
        plt.figure(figsize=(20, 10))
        ind = np.arange(len(scenarios))
        width = 0.6

        plt.bar(ind, total_travel_time, width, color='mediumpurple', label='Total Travel Time (hours)')

        if current_seoul_data and 'total_travel_time' in current_seoul_data:
            plt.axhline(y=current_seoul_data['total_travel_time'], color='red', linestyle='--', label='Current Seoul Travel Time Estimate')

        plt.ylabel('Total Travel Time (hours)')
        plt.xlabel('Scenario')
        plt.title(f'Total Travel Time by Scenario (Efficiency A = {a})')

        # X축 레이블을 10개만 표시
        plt.xticks(tick_indices, tick_labels, rotation=45)

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'total_travel_time_a_{a}.png'))
        plt.show()

        # 5. 평균 속도 시각화
        plt.figure(figsize=(20, 10))
        ind = np.arange(len(scenarios))
        width = 0.6

        plt.bar(ind, average_speed, width, color='teal', label=f'Average Speed (a={a})')

        if current_seoul_data and 'average_speed' in current_seoul_data:
            plt.axhline(y=current_seoul_data['average_speed'], color='red', linestyle='--', label='Current Seoul Average Speed')

        plt.ylabel('Average Speed (km/h)')
        plt.xlabel('Scenario')
        plt.title(f'Average Speed by Scenario for Efficiency A = {a}')
        
        # X축 레이블을 10개만 표시 (0%, 10%, ..., 100%)
        tick_indices = np.linspace(0, len(scenarios)-1, 11, dtype=int)
        tick_labels = [scenarios[i] for i in tick_indices]
        plt.xticks(tick_indices, tick_labels, rotation=45)
        
        plt.legend()
        plt.tight_layout()
        output_dir = 'figure'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'average_speed_a_{a}.png'))
        plt.show()

        # 6. 추가 에너지 소비 구성 요소 시각화
        plot_additional_energy_consumption(a_results, output_dir, scenarios, a, current_seoul_data)
        
        # 7. 에너지 소비 구성 비율 스택형 바차트
        plot_energy_consumption_breakdown(a_results, config, output_dir, scenarios, a)

        # 8. 시뮬레이션 결과의 분포 시각화
        plot_distribution(a_results, 'total_energy_electric', a)
        plot_distribution(a_results, 'total_carbon_electric', a)

def plot_additional_energy_consumption(results, output_dir, scenarios, a, current_seoul_data=None):
    """
    자율주행차 추가 에너지 소비 구성 요소를 시각화 (신뢰구간 포함)
    
    Parameters:
    - results: 시나리오 ���과가 담긴 딕셔너리의 리스트.
    - output_dir: 그래프를 저장할 디렉토리 경로 (str).
    - scenarios: 시나��오 리스트 (시각화용)
    - a: 효율성 계수
    - current_seoul_data: 현재 서울의 예상치를 담은 딕셔너리 (선택 사항).
    """
    E_compute_mean = [res['E_compute_mean'] for res in results]
    E_compute_ci_lower = [res['E_compute_ci_lower'] for res in results]
    E_compute_ci_upper = [res['E_compute_ci_upper'] for res in results]
    
    E_v2x_mean = [res['E_v2x_mean'] for res in results]
    E_v2x_ci_lower = [res['E_v2x_ci_lower'] for res in results]
    E_v2x_ci_upper = [res['E_v2x_ci_upper'] for res in results]
    
    E_data_center_mean = [res['E_data_center_mean'] for res in results]
    E_data_center_ci_lower = [res['E_data_center_ci_lower'] for res in results]
    E_data_center_ci_upper = [res['E_data_center_ci_upper'] for res in results]

    plt.figure(figsize=(20, 10))
    ind = np.arange(len(scenarios))
    width = 0.6

    # Compute Energy
    plt.bar(ind, E_compute_mean, width, label='Compute Energy (kWh)', color='teal')  # 기존 'purple'에서 'teal'으로 변경
    compute_errors = [
        [max(mean - lower, 0), max(upper - mean, 0)] 
        for mean, lower, upper in zip(E_compute_mean, E_compute_ci_lower, E_compute_ci_upper)
    ]
    plt.errorbar(ind, E_compute_mean, 
                 yerr=np.array(compute_errors).T,
                 fmt='none', ecolor='black', elinewidth=1, capsize=5)

    # V2X Energy
    plt.bar(ind, E_v2x_mean, width, bottom=E_compute_mean, label='V2X Energy (kWh)', color='magenta')  # 기존 'orange'에서 'magenta'으로 변경
    v2x_errors = [
        [max(mean - lower, 0), max(upper - mean, 0)] 
        for mean, lower, upper in zip(E_v2x_mean, E_v2x_ci_lower, E_v2x_ci_upper)
    ]
    plt.errorbar(ind, np.array(E_compute_mean) + np.array(E_v2x_mean), 
                 yerr=np.array(v2x_errors).T,
                 fmt='none', ecolor='black', elinewidth=1, capsize=5)

    # Data Center Energy
    bottom_E = np.array(E_compute_mean) + np.array(E_v2x_mean)
    plt.bar(ind, E_data_center_mean, width, bottom=bottom_E, label='Data Center Energy (kWh)', color='olive')  # 기존 'cyan'에서 'olive'으로 변경
    data_center_errors = [
        [max(mean - lower, 0), max(upper - mean, 0)] 
        for mean, lower, upper in zip(E_data_center_mean, E_data_center_ci_lower, E_data_center_ci_upper)
    ]
    plt.errorbar(ind, bottom_E + np.array(E_data_center_mean), 
                 yerr=np.array(data_center_errors).T,
                 fmt='none', ecolor='black', elinewidth=1, capsize=5)

    # 현재 서울의 추가 에너지 소비 예상치 기준선 추가
    current_seoul_data = {} if current_seoul_data is None else current_seoul_data

    if current_seoul_data:
        if 'additional_energy_electric' in current_seoul_data:
            plt.axhline(y=current_seoul_data['additional_energy_electric'], color='blue', linestyle='--', label='Current Seoul Additional Electric Energy')
        if 'additional_energy_gasoline' in current_seoul_data:
            plt.axhline(y=current_seoul_data['additional_energy_gasoline'], color='red', linestyle='--', label='Current Seoul Additional Gasoline Energy')

    plt.xlabel('Scenario')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title(f'Additional AV Energy Consumption Components by Scenario with 90% Confidence Interval (Efficiency A = {a})')

    # X축 레이블을 10개만 표시
    tick_indices = np.linspace(0, len(scenarios)-1, 11, dtype=int)
    tick_labels = [scenarios[i] for i in tick_indices]
    plt.xticks(tick_indices, tick_labels, rotation=45)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'additional_av_energy_consumption_a_{a}.png'))
    plt.show()

def plot_energy_consumption_breakdown(results, config, output_dir, scenarios, a):
    """
    시나리오별 차량 유형별 에너지 소비량을 시각화
    
    Parameters:
    - results: 시나리오 결과가 담긴 딕셔너리의 리스트.
    - output_dir: 그래프를 저장할 디렉토리 경로 (str).
    - scenarios: 시나리오 리스트 (시각화용)
    - a: 효율성 계수
    """
    clustering_value = config['general']['energy_reduction']['clustering']
    # 각 차량 유형별 에너지 소비량
    E_gasoline = 0
    E_electric = [res['E_electric_mean'] for res in results]
    E_drive_av = [res['E_drive_av_mean'] for res in results]
    E_compute = [res['E_compute_mean'] for res in results]
    E_v2x = [res['E_v2x_mean'] for res in results]
    E_data_center = [res['E_data_center_mean'] for res in results]
    E_RSU = [res['E_RSU_mean'] for res in results]

    # X축 및 그래프 설정
    ind = np.arange(len(scenarios))
    width = 0.6

    plt.figure(figsize=(20, 10))
    # Gasoline Energy
    plt.bar(ind, E_gasoline, width, label='Gasoline Vehicles (ℓ)', color='red')  # 기존 'lightcoral'에서 'red'으로 변경
    # Electric Energy
    plt.bar(ind, E_electric, width, bottom=E_gasoline, label='Electric Vehicles (kWh)', color='green')  # 기존 'skyblue'에서 'green'으로 변경
    bottom_E = np.array(E_gasoline) + np.array(E_electric)
    # AV Driving Energy
    plt.bar(ind, E_drive_av, width, bottom=bottom_E, label='AV Driving Energy (kWh)', color='blue')  # 기존 'steelblue'에서 'blue'으로 변경
    bottom_E += np.array(E_drive_av)
    # Compute Energy
    plt.bar(ind, E_compute, width, bottom=bottom_E, label='AV Compute Energy (kWh)', color='orange')  # 기존 'dodgerblue'에서 'orange'으로 변경
    bottom_E += np.array(E_compute)
    # V2X Energy
    plt.bar(ind, E_v2x, width, bottom=bottom_E, label='AV V2X Energy (kWh)', color='purple')  # 기존 'deepskyblue'에서 'purple'으로 변경
    bottom_E += np.array(E_v2x)
    # Data Center Energy
    plt.bar(ind, E_data_center, width, bottom=bottom_E, label='AV Data Center Energy (kWh)', color='brown')  # 기존 'lightskyblue'에서 'brown'으로 변경
    bottom_E += np.array(E_data_center)
    # RSU Energy
    plt.bar(ind, E_RSU, width, bottom=bottom_E, label='RSU Energy (kWh)', color='grey')  # 기존 'royalblue'에서 'grey'으로 변경

    # X축 간소화 (대표값만 표시)
    tick_indices = np.linspace(0, len(scenarios)-1, 11, dtype=int)
    tick_labels = [scenarios[i] for i in tick_indices]
    plt.xticks(tick_indices, tick_labels, rotation=45)

    # 그래프 레이블 및 타이틀 설정
    plt.ylabel('Energy Consumption (kWh)')
    plt.xlabel('Scenario')
    plt.title(f'Energy Consumption Breakdown by Scenario (Efficiency A = {a})')
    plt.legend()

    # 그래프 저장 및 표시
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'energy_consumption_breakdown_a_{a}.png'))
    plt.show()

def plot_distribution(results, parameter, a, output_dir='figure'):
    """
    시뮬레이션 결과의 분포를 시각화하는 함수
    Parameters:
    - results: 시뮬레이션 결과 리스트
    - parameter: 분포를 시각화할 파라미터 (str)
    - a: 효율성 계수
    - output_dir: 그래프를 저장할 디렉토리 경로 (str)
    """
    data = [res[f"{parameter}_mean"] for res in results]
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {parameter} (Efficiency A = {a})')
    plt.xlabel(parameter)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{parameter}_distribution_a_{a}.png'))
    plt.show()

# 시각화 코드는 변경된 데이터 구조를 처리할 수 있도록 이미 수정되어 있다고 가정합니다.
# 만약 필요하다면, 벡터화된 데이터에 맞게 수정합니다.
