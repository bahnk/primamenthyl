{{ config(tags=['individual']) }}

select
    samples.sample,
    avg(record_counts.methylation_count) as mean_methylation_count
from {{ source('quant', 'quant__samples') }} as samples
cross join {{ ref('stg_record_methylation_counts') }} as record_counts
group by samples.sample
