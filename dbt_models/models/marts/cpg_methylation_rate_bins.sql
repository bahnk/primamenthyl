{{ config(tags=['individual']) }}

select
    samples.sample,
    floor(positions.position / 100000) * 100000 as bin,
    cast(sum(case when positions.methylated then 1 else 0 end) as double)
    / nullif(count(*), 0) as cpg_methylation_rate
from {{ source('quant', 'quant__samples') }} as samples
cross join {{ ref('stg_methylation_positions') }} as positions
where positions.reference = 'chr21'
    and positions.context = 'cpg'
group by samples.sample, bin
