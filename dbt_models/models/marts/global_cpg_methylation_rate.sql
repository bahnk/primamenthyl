{{ config(tags=['individual']) }}

select
    samples.sample,
    cast(cpg_methylated.methylated_count as double)
    / nullif(cpg_methylated.methylated_count + cpg_unmethylated.unmethylated_count, 0)
        as global_cpg_methylation_rate
from {{ source('quant', 'quant__samples') }} as samples
cross join (
    select count(*) as methylated_count
    from {{ source('quant', 'quant__cpg_methylated') }}
) as cpg_methylated
cross join (
    select count(*) as unmethylated_count
    from {{ source('quant', 'quant__cpg_unmethylated') }}
) as cpg_unmethylated
