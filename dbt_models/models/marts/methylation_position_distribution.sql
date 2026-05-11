{{ config(materialized='table', tags=['merged']) }}

select
    records.sample,
    records.reference,
    methylation.context,
    methylation.methylated,
    floor((records.position + methylation.methylation_offset) / 100000) * 100000 as position_bin,
    count(*) as methylation_count
from {{ ref('stg_methylation_events') }} as methylation
inner join {{ source('quant', 'quant__records') }} as records
    on methylation.sample = records.sample
    and methylation.align_id = records.align_id
group by records.sample, records.reference, methylation.context, methylation.methylated, position_bin
