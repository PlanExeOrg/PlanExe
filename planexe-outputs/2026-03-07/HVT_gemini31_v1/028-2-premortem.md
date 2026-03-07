A premortem assumes the project has failed and works backward to identify the most likely causes.

## Assumptions to Kill

These foundational assumptions represent the project's key uncertainties. If proven false, they could lead to failure. Validate them immediately using the specified methods.

| ID | Assumption | Validation Method | Failure Trigger |
|----|------------|-------------------|-----------------|
| A1 | Specialty insurance for autonomous drone-human interaction is available at a cost below 15% of revenue. | Submit the full technical safety dossier and CONOPS to three specialty aviation brokers for a binding quote. | All brokers decline to quote or provide premiums exceeding 20% of projected annual revenue. |
| A2 | Private 5G/CBRS networks can maintain sub-20ms latency in urban environments with metallic obstacles. | Conduct a 3D ray-tracing propagation study and on-site RF spectrum analysis at the primary target site. | Simulated or measured latency consistently exceeds 20ms due to multipath interference from arena structures. |
| A3 | Hardware-level proximity sensors and propeller shrouds satisfy FSDO safety requirements for 'Operation Over Human Beings'. | Schedule a pre-application meeting with the local FSDO to present the safety case and hardware specifications. | FSDO indicates that hardware-level sensors are insufficient and requires a formal Part 107 waiver with additional kinetic energy impact testing. |
| A4 | The target demographic in Austin, LA, and RTP will sustain a 40% repeat-customer rate through tiered memberships. | Run a 'shadow-pricing' survey and a limited-time beta event to measure actual interest and price sensitivity. | Survey results indicate a repeat-customer intent of less than 25%. |
| A5 | The proprietary 'soft-drone' chassis and infrared tagging system will survive 50+ hours of high-speed urban-mimicry combat without critical failure. | Conduct a 100-hour stress test of the drone fleet in a simulated arena environment with high-impact obstacles. | Mean Time Between Failures (MTBF) drops below 20 flight hours during the stress test. |
| A6 | Corporate team-building clients will view the HVT simulation as a high-value leadership training tool rather than a generic recreational activity. | Conduct structured interviews with five corporate event planners to validate the curriculum and value proposition. | Corporate planners indicate the experience lacks the professional development depth required for their training budgets. |
| A7 | The local neighborhood councils will view the HVT arena as a tech-forward recreational asset rather than a noise-polluting nuisance. | Conduct a mock community open day and decibel-level test at the proposed site perimeter. | Community feedback indicates a 60%+ disapproval rating or noise complaints exceed local ordinance limits. |
| A8 | The 'HVT Social-Combat App' will generate sufficient viral loops to lower customer acquisition costs (CAC) by 20% within 6 months. | Launch a prototype version of the app with a limited user group to track share-to-booking conversion rates. | The viral coefficient (K-factor) is less than 1.0, indicating no organic growth. |
| A9 | The supply chain for industrial-grade drone components is resilient enough to maintain a 20% spare parts buffer without exceeding lead times of 14 days. | Place a test order for critical components (motors, flight controllers) with all primary and secondary vendors. | Vendor lead times exceed 21 days or critical components are backordered indefinitely. |


## Failure Scenarios and Mitigation Plans

Each scenario below links to a root-cause assumption and includes a detailed failure story, early warning signs, measurable tripwires, a response playbook, and a stop rule to guide decision-making.

### Summary of Failure Modes

| ID | Title | Archetype | Root Cause | Owner | Risk Level |
|----|-------|-----------|------------|-------|------------|
| FM1 | The Insolvency Trap | Process/Financial | A1 | Risk Management & Insurance Liaison | CRITICAL (20/25) |
| FM2 | The Latency Blackout | Technical/Logistical | A2 | Technical Lead | CRITICAL (15/25) |
| FM3 | The Regulatory Shutdown | Market/Human | A3 | Aviation Regulatory & Compliance Officer | CRITICAL (20/25) |
| FM4 | The Novelty Decay Spiral | Process/Financial | A4 | Business Development & Corporate Sales Lead | CRITICAL (16/25) |
| FM5 | The Maintenance Bottleneck | Technical/Logistical | A5 | Drone Systems & Robotics Engineer | HIGH (12/25) |
| FM6 | The Corporate Value Gap | Market/Human | A6 | Business Development & Corporate Sales Lead | HIGH (12/25) |
| FM7 | The Zoning Backlash | Process/Financial | A7 | Community & Public Relations Coordinator | CRITICAL (15/25) |
| FM8 | The Viral Void | Technical/Logistical | A8 | Product Lead | HIGH (12/25) |
| FM9 | The Supply Chain Collapse | Market/Human | A9 | Drone Systems & Robotics Engineer | HIGH (12/25) |


### Failure Modes

#### FM1 - The Insolvency Trap

- **Archetype**: Process/Financial
- **Root Cause**: Assumption A1
- **Owner**: Risk Management & Insurance Liaison
- **Risk Level:** CRITICAL 20/25 (Likelihood 4/5 × Impact 5/5)

##### Failure Story
The project relies on a 15% insurance cost threshold to maintain profitability. If premiums spike due to the high-risk nature of autonomous drone-human combat, the operating margin collapses. Without a binding quote, the $450,000 seed capital is deployed into a business model that cannot legally or financially sustain itself, leading to a total loss of investment when the insurance carrier denies coverage post-launch.

##### Early Warning Signs
- Brokers request additional safety data beyond the initial dossier
- Initial quotes exceed 18% of projected revenue
- Underwriters express concern over 'intentional contact' clauses

##### Tripwires
- Insurance premium quote > 20% of revenue
- Days without binding insurance quote > 90

##### Response Playbook
- Contain: Immediately freeze all non-essential capital expenditure.
- Assess: Re-calculate ROI based on a 25% insurance cost scenario.
- Respond: Pivot to a 'tethered' or 'simulated-only' drone model to reduce liability profile.


**STOP RULE:** If a binding insurance quote is not secured within 120 days, the project is cancelled.

---

#### FM2 - The Latency Blackout

- **Archetype**: Technical/Logistical
- **Root Cause**: Assumption A2
- **Owner**: Technical Lead
- **Risk Level:** CRITICAL 15/25 (Likelihood 3/5 × Impact 5/5)

##### Failure Story
Urban RF noise and signal reflections from metallic shipping containers create 'dead zones' in the arena. When the drone swarm-AI loses its C2 link, it fails to execute the 'Hover-and-Land' protocol, resulting in a high-speed collision with a player. This technical failure leads to immediate fleet grounding by the FSDO and permanent loss of the facility's operational license.

##### Early Warning Signs
- Packet loss rates > 2% during bench testing
- Jitter measurements exceeding 15ms in simulated urban environments
- Signal propagation gaps identified in 3D ray-tracing models

##### Tripwires
- Average latency >= 25ms in arena test
- Packet loss >= 5% during peak load simulation

##### Response Playbook
- Contain: Ground the entire drone fleet immediately.
- Assess: Perform a full RF spectrum audit to identify interference sources.
- Respond: Deploy additional CBRS small-cell nodes and reconfigure network topology.


**STOP RULE:** If sub-20ms latency cannot be maintained in a full-scale arena test, the project pivots to manual-only flight.

---

#### FM3 - The Regulatory Shutdown

- **Archetype**: Market/Human
- **Root Cause**: Assumption A3
- **Owner**: Aviation Regulatory & Compliance Officer
- **Risk Level:** CRITICAL 20/25 (Likelihood 4/5 × Impact 5/5)

##### Failure Story
The project assumes that hardware-level safety features are sufficient to satisfy the FSDO. However, the FSDO classifies the arena as a high-risk aviation site rather than a standard recreational facility. The lack of a formal Part 107 waiver for 'Operation Over Human Beings' leads to a surprise inspection, a cease-and-desist order, and a public relations crisis that destroys the brand's reputation before it can scale.

##### Early Warning Signs
- FSDO requests for additional safety documentation exceed 3 iterations
- Local zoning board expresses concern over 'military-style' drone activity
- Negative media coverage regarding drone privacy in the target city

##### Tripwires
- FSDO response time > 60 days
- Permit denial count = 1

##### Response Playbook
- Contain: Cease all autonomous flight operations immediately.
- Assess: Engage external aviation counsel to review the FSDO's enforcement stance.
- Respond: Transition to 'pilot-in-command supervised' operations to maintain revenue while pursuing a formal waiver.


**STOP RULE:** If the FSDO issues a formal cease-and-desist order, the project is suspended until a full waiver is granted.

---

#### FM4 - The Novelty Decay Spiral

- **Archetype**: Process/Financial
- **Root Cause**: Assumption A4
- **Owner**: Business Development & Corporate Sales Lead
- **Risk Level:** CRITICAL 16/25 (Likelihood 4/5 × Impact 4/5)

##### Failure Story
The business model relies on high customer lifetime value (CLV) to offset the $450,000 capital expenditure. If the experience fails to retain players, the CAC-to-CLV ratio exceeds 1:2, leading to cash flow insolvency. The high fixed costs of the arena cannot be covered by one-time visitors, and the lack of a 'killer app' viral loop prevents organic growth, forcing the project into a permanent state of negative cash flow.

##### Early Warning Signs
- Repeat-customer rate < 20% after 3 months
- Customer acquisition costs increasing by > 15% month-over-month

##### Tripwires
- Monthly churn rate >= 60%
- CLV/CAC ratio <= 1.5

##### Response Playbook
- Contain: Immediately reduce marketing spend to preserve remaining cash.
- Assess: Survey churned customers to identify the primary reason for non-return.
- Respond: Pivot the facility to a 'Drone Flight Academy' model to capture stable, recurring educational revenue.


**STOP RULE:** If the break-even point extends beyond 30 months, the project is cancelled.

---

#### FM5 - The Maintenance Bottleneck

- **Archetype**: Technical/Logistical
- **Root Cause**: Assumption A5
- **Owner**: Drone Systems & Robotics Engineer
- **Risk Level:** HIGH 12/25 (Likelihood 3/5 × Impact 4/5)

##### Failure Story
The proprietary 'soft-drone' hardware is too fragile for the high-impact urban-mimicry environment. Frequent collisions cause structural fatigue in the carbon-fiber frames and sensor mounts. The 24-hour repair cycle is overwhelmed by the volume of failures, leading to a fleet availability rate below 70%. This downtime causes missed bookings, contract penalties for corporate events, and a total collapse of the operational throughput required for profitability.

##### Early Warning Signs
- Mean Time Between Failures (MTBF) < 20 flight hours
- Maintenance labor hours exceeding 1.5 hours per flight hour

##### Tripwires
- Fleet availability <= 70%
- Unplanned maintenance downtime > 48 hours per week

##### Response Playbook
- Contain: Reduce daily session capacity to match current fleet availability.
- Assess: Perform a root-cause analysis on the most frequent failure points in the drone chassis.
- Respond: Pivot to off-the-shelf, ruggedized consumer drone frames to ensure fleet reliability.


**STOP RULE:** If fleet availability remains below 70% for 30 consecutive days, the project is suspended for hardware redesign.

---

#### FM6 - The Corporate Value Gap

- **Archetype**: Market/Human
- **Root Cause**: Assumption A6
- **Owner**: Business Development & Corporate Sales Lead
- **Risk Level:** HIGH 12/25 (Likelihood 4/5 × Impact 3/5)

##### Failure Story
The project fails to secure high-margin corporate team-building contracts because the simulation is perceived as 'just a game' rather than a professional development tool. Without these contracts, the revenue model relies entirely on low-margin, high-churn recreational sessions. The inability to reach the 5-contract-per-month target leaves the project unable to cover the high overhead of the purpose-built arena, leading to a failure to break even within the 18-month window.

##### Early Warning Signs
- Corporate sales pipeline conversion rate < 5%
- Feedback from corporate clients citing lack of 'professional value'

##### Tripwires
- Corporate revenue < 10% of total monthly revenue
- Number of corporate contracts = 0 for 3 consecutive months

##### Response Playbook
- Contain: Halt all outbound sales efforts to corporate clients to avoid brand damage.
- Assess: Conduct a deep-dive interview with lost corporate leads to identify the 'value gap'.
- Respond: Rebrand the experience with a formal 'Leadership & Communication' curriculum to align with corporate training needs.


**STOP RULE:** If corporate revenue does not reach 25% of total revenue by month 12, the project is pivoted to a pure-play recreational model.

---

#### FM7 - The Zoning Backlash

- **Archetype**: Process/Financial
- **Root Cause**: Assumption A7
- **Owner**: Community & Public Relations Coordinator
- **Risk Level:** CRITICAL 15/25 (Likelihood 3/5 × Impact 5/5)

##### Failure Story
The project relies on a favorable reception from local neighborhood councils to maintain its recreational zoning status. If the community perceives the arena as a noisy, privacy-invading 'military-style' operation, they will lobby the municipal board to revoke the facility's permits. This leads to a forced relocation, incurring a 25% budget increase for site preparation and a 6-month operational delay, effectively bankrupting the venture before it achieves scale.

##### Early Warning Signs
- Negative media coverage in local neighborhood newsletters
- Attendance at municipal zoning hearings by opposition groups > 20 people

##### Tripwires
- Community disapproval rating >= 50%
- Noise complaints > 5 per week

##### Response Playbook
- Contain: Immediately halt all outdoor flight operations during evening hours.
- Assess: Commission an independent decibel-level audit to verify compliance.
- Respond: Initiate a formal community outreach program to rebrand the facility as a 'STEM Education & Drone Flight Academy'.


**STOP RULE:** If the municipal board issues a formal notice of zoning violation, the project is cancelled.

---

#### FM8 - The Viral Void

- **Archetype**: Technical/Logistical
- **Root Cause**: Assumption A8
- **Owner**: Product Lead
- **Risk Level:** HIGH 12/25 (Likelihood 4/5 × Impact 3/5)

##### Failure Story
The project's financial model assumes that the 'HVT Social-Combat App' will drive viral growth and lower CAC. If the app fails to gain traction, the project is forced to rely on expensive, traditional paid advertising. This increases CAC by 40%, causing the CAC-to-CLV ratio to exceed 1:2. The resulting financial strain prevents the 10% revenue reinvestment into R&D, leading to a rapid decline in player engagement and eventual insolvency.

##### Early Warning Signs
- App download rate < 10% of total site visitors
- Social media share rate < 5% per session

##### Tripwires
- Viral coefficient (K-factor) < 0.5
- CAC/CLV ratio > 1:2

##### Response Playbook
- Contain: Suspend all app development costs to preserve cash.
- Assess: Conduct A/B testing on app features to identify the friction points in the sharing loop.
- Respond: Pivot marketing strategy to focus on direct corporate partnerships to bypass the need for viral growth.


**STOP RULE:** If the app fails to achieve a K-factor of 1.0 within 6 months, the project is pivoted to a non-viral, direct-sales model.

---

#### FM9 - The Supply Chain Collapse

- **Archetype**: Market/Human
- **Root Cause**: Assumption A9
- **Owner**: Drone Systems & Robotics Engineer
- **Risk Level:** HIGH 12/25 (Likelihood 3/5 × Impact 4/5)

##### Failure Story
The project's reliance on specialized industrial-grade drone components makes it highly vulnerable to supply chain disruptions. If primary and secondary vendors fail to meet the 14-day lead time, the 20% spare parts buffer is depleted within 3 weeks. The resulting fleet downtime (exceeding 30%) leads to massive booking cancellations and a loss of brand credibility, causing the 40% repeat-customer rate to plummet as players lose confidence in the facility's reliability.

##### Early Warning Signs
- Vendor lead times increasing by > 5 days
- Inventory levels of critical components falling below 10% buffer

##### Tripwires
- Vendor lead time > 21 days
- Fleet availability < 60%

##### Response Playbook
- Contain: Immediately source off-the-shelf consumer drone components as a temporary stop-gap.
- Assess: Audit all vendor contracts to identify the source of the supply chain failure.
- Respond: Diversify the vendor base to include at least three geographically distinct suppliers for all critical components.


**STOP RULE:** If fleet availability remains below 60% for 14 consecutive days due to supply chain issues, the project is suspended.
