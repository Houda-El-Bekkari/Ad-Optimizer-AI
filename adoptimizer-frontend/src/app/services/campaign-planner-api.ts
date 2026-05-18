import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';

export interface CampaignPlannerRequest {
  objectif: string;
  budget: number;
  plateforme: string;
  produit: string;
}

export interface CampaignPlannerKpis {
  roas_j14?: number;
  conversions_j14?: number;
  cpa_j14?: number;
  ctr_j14?: number;
  cpc_j14?: number;
  [key: string]: unknown;
}

export interface CampaignPlannerRecommendation {
  strategy_id?: string;
  name?: string;
  type?: string;
  platform?: string;
  objective?: string;
  product?: string;
  budget?: number;
  kpis?: CampaignPlannerKpis;
  targets?: Record<string, unknown>;
}

export interface CampaignPlannerChannelPlan {
  platform?: string;
  strategy_id?: string;
  name?: string;
  type?: string;
  budget?: number;
  score_final?: number;
  kpis?: CampaignPlannerKpis;
  targets?: Record<string, unknown>;
}

export interface CampaignPlannerMultiPlatformPlan {
  mode?: string;
  total_budget?: number;
  allocated_budget?: number;
  reserve_budget?: number;
  channels?: CampaignPlannerChannelPlan[];
}

export interface CampaignPlannerActionPlanItem {
  step?: number;
  title?: string;
  action?: string;
}

export interface CampaignPlannerConfidence {
  level?: string;
  score?: number;
  reasons?: string[];
}

export interface CampaignPlannerResponse {
  title?: string;
  generated_at?: string;
  generation_mode?: string;
  final_message?: string;
  recommendation?: CampaignPlannerRecommendation;
  best_by_platform?: Record<string, CampaignPlannerChannelPlan>;
  multi_platform_plan?: CampaignPlannerMultiPlatformPlan | null;
  why?: string[];
  multi_platform_why?: string[];
  risks?: string[];
  action_plan?: CampaignPlannerActionPlanItem[];
  multi_platform_action_plan?: CampaignPlannerActionPlanItem[];
  confidence?: CampaignPlannerConfidence;
}

@Injectable({ providedIn: 'root' })
export class CampaignPlannerApi {
  private readonly webhookUrl =
    `${environment.apiUrl}/workflows/case2/campaign`;

  constructor(private readonly http: HttpClient) {}

  generateStrategy(payload: CampaignPlannerRequest): Observable<CampaignPlannerResponse> {
    return this.http.post<CampaignPlannerResponse>(this.webhookUrl, payload);
  }
}
